#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <boost/foreach.hpp>

#include <iostream>
#include <stdlib.h>

using namespace message_filters;

Eigen::MatrixXf P;

void callback(const sensor_msgs::ImageConstPtr &msg_img, const sensor_msgs::PointCloud2ConstPtr &msg_pointcloud)
{
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8);
	cv::Mat frame_rgb = cv_ptr->image;

	pcl::PointCloud<pcl::PointXYZ> laser;
	pcl::PointCloud<pcl::PointXYZ> visible_laser;
	pcl::fromROSMsg(*msg_pointcloud, laser);
	

	//相机可视点云
	for(int i = 0; i < laser.points.size(); i ++)
	{
		pcl::PointXYZ point = laser.points[i];
		if(point.z < 0)
		{
			continue;
		}

		Eigen::Vector3f point_v(point.x, point.y, point.z);
		Eigen::Vector3f rgb = P * point_v;

		if(rgb(0) / rgb(2) < frame_rgb.rows && rgb(1) / rgb(2) < frame_rgb.cols)
		{
			visible_laser.push_back(point);
		}
	}

	//提取包含4个圆的平面点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr visible_laser_ptr(new pcl::PointCloud<pcl::PointXYZ>(visible_laser));
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(visible_laser_ptr));
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
	ransac.setDistanceThreshold(0.01);
	ransac.computeModel();

	pcl::PointCloud<pcl::PointXYZ> plane;
	std::vector<int> inliers_indicies;
	ransac.getInliers(inliers_indicies);
	pcl::copyPointCloud<pcl::PointXYZ>(visible_laser, inliers_indicies, plane);

	//提取4个圆心
	std::vector<cv::Point3f> pts_3d;
	std::vector<float> radius;
	for(int i = 0; i < 4; i ++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr plane_ptr(new pcl::PointCloud<pcl::PointXYZ>(plane));
		pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(plane_ptr));
		model_s->setRadiusLimits(0.05, 0.07);
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac_sphere(model_s);
		ransac_sphere.setDistanceThreshold(0.005);
		ransac_sphere.computeModel();

		inliers_indicies.clear();
		ransac_sphere.getInliers(inliers_indicies);

		Eigen::VectorXf coeficients;
		ransac_sphere.getModelCoefficients(coeficients);

		pts_3d.push_back(cv::Point3f(coeficients(0), coeficients(1), coeficients(2)));
		radius.push_back(coeficients(3));

		pcl::PointCloud<pcl::PointXYZ> outliers;
		for(int i = 0; i < plane.size(); i ++)
		{
			std::vector<int>::iterator iter = find(inliers_indicies.begin(), inliers_indicies.end(), i);
			if(iter == inliers_indicies.end())
			{
				outliers.push_back(plane[i]);
			}
		}

		plane.clear();
		plane = outliers;
	}


	//霍夫变换找圆
	cv::Mat gray;
	cv::cvtColor(frame_rgb, gray, CV_BGR2GRAY);
	std::vector<cv::Vec3f> circles;
	int canny_thresh = 150;
	int center_thresh = 80;
	for(int thresh = center_thresh; circles.size() < 4 && thresh > 30; thresh -= 5)
	{
		cv::HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows / 5, canny_thresh, thresh, 0, 0);
	}
	std::vector<cv::Point2f> pts_2d;
	for(int i = 0; i < circles.size(); i ++)
	{
		pts_2d.push_back(cv::Point2f(circles[i](0), circles[i](1)));
	}


	cv::Mat R, T, K;
	cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), R, T, false, cv::SOLVEPNP_EPNP);

	std::cout << R << std::endl;
	std::cout << T << std::endl;
}



int main(int argc, char **argv)
{
	ros::init(argc, argv, "calibration");
	ros::NodeHandle nh;

	message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera", 1);
	message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, "/velodyne_points", 1);

	TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> sync(image_sub, cloud_sub, 10);       // 同步
	sync.registerCallback(boost::bind(&callback, _1, _2));                   // 回调

	//typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
	//Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, cloud_sub);
	//sync.registerCallback(boost::bind(&callback, _1, _2));

	ros::spin();
	return EXIT_SUCCESS;
}
