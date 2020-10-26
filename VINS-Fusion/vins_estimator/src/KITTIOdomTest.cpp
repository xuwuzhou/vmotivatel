/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "estimator/parameters.h"
#include "estimator/estimator.h"
#include "utility/visualization.h"

#include "common.h"
#include "lidarFactor.hpp"

#define DISTORTION 0

using namespace std;
using namespace Eigen;
int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

//跳帧数，控制发给laserMapping的频率
int skipFrameNum = 5;
bool systemInited = false;

//时间戳信息
double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;



pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());//receive sharp points
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());//receive less sharp points
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());//receive flat points
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());//receive less flat points

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());//less sharp points of last frame
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());//less flat points of last frame
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());//receive all points

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;




Estimator estimator;

Eigen::Matrix3d c1Rc0, c0Rc1;
Eigen::Vector3d c1Tc0, c0Tc1;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
queue<sensor_msgs::PointCloud2ConstPtr> depth_cloudBuf;
pcl::PointCloud<pcl::PointXYZI>::Ptr countCloud(new pcl::PointCloud<pcl::PointXYZI>());
std::mutex m_buf;
int num,num0,num1; 
int num3,num4;
int System_count = 0;
bool System_inited =false;

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// transform all lidar points to the start of the next frame
//将上一帧点云中的点相对结束位置去除因匀速运动产生的畸变，效果相当于得到在点云扫描结束位置静止扫描得到的点云
//void TransformToEnd(PointType const *const pi, PointType *const po)
//{
//    // undistort point first
//    pcl::PointXYZI un_point_tmp;
//    TransformToStart(pi, &un_point_tmp);

//    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
//    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

//    po->x = point_end.x();
//    po->y = point_end.y();
//    po->z = point_end.z();

//    //Remove distortion time info
//    po->intensity = int(pi->intensity);
//}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    m_buf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
//    pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
//    pcl::fromROSMsg(*cornerPointsSharp2,*tmpCloud);
//    cout<<"kitti odometry  cornerPointsSharp num:"<<tmpCloud->points.size()<<endl;
    m_buf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    m_buf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
//    pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
//    pcl::fromROSMsg(*cornerPointsLessSharp2,*tmpCloud);
//    cout<<"kitti odometry  cornerPointsLessSharp num:"<<tmpCloud->points.size()<<endl;
    m_buf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    m_buf.lock();
    surfFlatBuf.push(surfPointsFlat2);
//    pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
//    pcl::fromROSMsg(*surfPointsFlat2,*tmpCloud);
//    cout<<"kitti odometry  surfPointsFlat2 num:"<<tmpCloud->points.size()<<endl;
    m_buf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    m_buf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
//    pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
//    pcl::fromROSMsg(*surfPointsLessFlat2,*tmpCloud);
//    cout<<"kitti odometry  surfPointsLessFlat2 num:"<<tmpCloud->points.size()<<endl;
    m_buf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    m_buf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
//    pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
//    pcl::fromROSMsg(*laserCloudFullRes2,*tmpCloud);
//    cout<<"kitti odometry  laserCloudFullRes2 num:"<<tmpCloud->points.size()<<endl;
    m_buf.unlock();
}

void depthCloud_callback(const sensor_msgs::PointCloud2ConstPtr &depth_cloud)
{
	if(!System_inited){
		System_count++;
		if(System_count>=21){System_inited = true;}
		else{return;}
	}
	m_buf.lock();
	if(depth_cloud==NULL){num3++;printf("深度图%d空指针",num3);}
	else {num4++;printf("指针%d非空",num4);}
	if(depth_cloud!=NULL)
	depth_cloudBuf.push(depth_cloud);
	if(depth_cloudBuf.empty())printf("深度图队列空");
	m_buf.unlock();
}

void callback(const sensor_msgs::ImageConstPtr &img_msg0,const sensor_msgs::ImageConstPtr &img_msg1)
{ 
    estimator.inputImage(img_msg0->header.stamp.toSec(),getImageFromMsg(img_msg0),getImageFromMsg(img_msg1));
    num++;
    printf("处理第%d帧",num);

}


void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
    num0++;
    printf("缓存队列加入了第%d左帧\n",num0);
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
    num1++;
    printf("缓存队列加入了第%d右帧\n",num1);
}


// extract images with same timestamp from two topics
void sync_process()
{
    while(1)
    {
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
	    sensor_msgs::PointCloud2ConstPtr depthCloudtmp(new sensor_msgs::PointCloud2);
	    sensor_msgs::PointCloud2ConstPtr cornerPointsSharpSensor(new sensor_msgs::PointCloud2);
	    sensor_msgs::PointCloud2ConstPtr surfPointsFlatSensor(new sensor_msgs::PointCloud2);
	    sensor_msgs::PointCloud2ConstPtr cornerPointsLessSharpSensor(new sensor_msgs::PointCloud2);
	    sensor_msgs::PointCloud2ConstPtr surfPointsLessFlatSensor(new sensor_msgs::PointCloud2);
	    sensor_msgs::PointCloud2ConstPtr fullPointSensor(new sensor_msgs::PointCloud2);
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
               
                
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
//		    pcl::fromROSMsg(*depthCloudtmp,*countCloud);
		    pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
		    if(System_count>=21&&depth_cloudBuf.size()>=1){
		    depthCloudtmp = depth_cloudBuf.front();
		    pcl::fromROSMsg(*depthCloudtmp,laserCloudIn);
		    depth_cloudBuf.pop();
		    printf("在主进程接收到深度图点云后开始计算点云大小");
		    }
		    else{
			printf("在主进程接收到深度图点云后因为不符合条件，所以不进行类型转换");
		    }
		    
		   
		    
                    //printf("find img0 and img1\n");
		   
		   
		    printf("该深度图点云的点的数量%d\n",laserCloudIn.points.size());
                
            }
	if(!cornerSharpBuf.empty() && !surfFlatBuf.empty() && !cornerLessSharpBuf.empty() && !surfLessFlatBuf.empty() && !fullPointsBuf.empty()&&
        !image0.empty())
	{	
	     //将激光点传入estimator 相机 yiqi jin
		            cornerPointsSharp->clear();
            	    pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
		            cornerPointsSharpSensor=cornerSharpBuf.front();
		            cout<<"kitti odometry cornerPointsSharp 2: "<<cornerPointsSharp->points.size()<<endl;
            	    cornerSharpBuf.pop();

            	    cornerPointsLessSharp->clear();
            	    pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            	    cout<<"kitti odometry cornerLessSharp 2: "<<cornerPointsLessSharp->points.size()<<endl;
            	    cornerLessSharpBuf.pop();

            	    surfPointsFlat->clear();
            	    pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
		            surfPointsFlatSensor = surfFlatBuf.front();
		            cout<<"kitti odometry surfFlat 2:"<<surfPointsFlat->points.size()<<endl;
            	    surfFlatBuf.pop();

            	    surfPointsLessFlat->clear();
            	    pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            	    cout<<"kitti odometry surfLessFlat 2: "<<surfPointsLessFlat->points.size()<<endl;
            	    surfLessFlatBuf.pop();

            	    laserCloudFullRes->clear();
            	    pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            	    cout<<"kitti odometry fullPoints: "<<laserCloudFullRes->points.size()<<endl;
            	    fullPointsBuf.pop();
            	    if(num>=11)
            	    estimator.inputLidarPoints(cornerPointsSharp,surfPointsFlat,cornerPointsLessSharp,surfPointsLessFlat,laserCloudFullRes);

            	    printf("将第%d左右帧送入里程计处理\n",num);
	}
            m_buf.unlock();
	    
            if(!image0.empty()){
                estimator.inputImage(time, image0, image1);
                num++;
                printf("将第%d左右帧送入里程计处理\n",num);}
		//estimator.depthCloudproj(depth_cloudBuf.front());
		//depth_cloudBuf.pop();
//		int result = (cornerPointsLessSharpSensor==NULL)?1:0;
		//cout<<"cornerpointsensor is NULL ,if NULL =1, if NONULL =0: the result ="<<result<<endl; //这一行没有用，测不出来空
//		pcl::PointCloud<PointType>::Ptr tmpcornerPointsSharp(new pcl::PointCloud<PointType>());
//		pcl::fromROSMsg(*cornerPointsSharpSensor,*tmpcornerPointsSharp);
//		pcl::PointCloud<PointType>::Ptr tmpcornerPointsLessSharp(new pcl::PointCloud<PointType>());
//		pcl::fromROSMsg(*cornerPointsLessSharpSensor,*tmpcornerPointsLessSharp);
//		pcl::PointCloud<PointType>::Ptr tmpsurfPointsFlat(new pcl::PointCloud<PointType>());
//		pcl::fromROSMsg(*surfPointsFlatSensor,*tmpsurfPointsFlat);
//		pcl::PointCloud<PointType>::Ptr tmpsurfPointsLessFlat(new pcl::PointCloud<PointType>());
//		pcl::fromROSMsg(*surfPointsLessFlatSensor,*tmpsurfPointsLessFlat);
//		pcl::PointCloud<PointType>::Ptr tmpfullpoints(new pcl::PointCloud<PointType>());
//		pcl::fromROSMsg(*fullPointSensor,*tmpfullpoints);
//		cout<<"tmpcornerPointsSharp size :"<<tmpcornerPointsSharp->points.size()<<endl;
//		cout<<"tmpcornerPointsLessSharp size :"<<tmpcornerPointsLessSharp->points.size()<<endl;
//		cout<<"tmpsurfPointsFlat size :"<<tmpsurfPointsFlat->points.size()<<endl;
//		cout<<"tmpsurfPointsLessFlat size :"<<tmpsurfPointsLessFlat->points.size()<<endl;
//		cout<<"tmpfullpoints size :"<<tmpfullpoints->points.size()<<endl;

//		if(tmpcornerPointsSharp->points.size()!=0&&tmpcornerPointsLessSharp->points.size()!=0&&
//		tmpsurfPointsFlat->points.size()!=0&&tmpsurfPointsLessFlat->points.size()!=0&&
//		tmpfullpoints->points.size()!=0)
//		{
//////			printf("验证输入条件满足\n");
//////			estimator.depthCloudproj(depthCloudtmp,cornerPointsSharpSensor,surfPointsFlatSensor);
//	//		    estimator.inputLidarPoints(cornerPointsSharpSensor,surfPointsFlatSensor,cornerPointsLessSharpSensor,surfPointsLessFlatSensor,fullPointSensor);
//		}
        }
        else
        {

        }

        
    }
}


int main(int argc, char** argv)
{
        num=0;
        num0=0;
        num1=0;
	ros::init(argc, argv, "vins_estimator");
	ros::NodeHandle n("~");
	ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

	


	if(argc != 2)
	{
		printf("please intput: rosrun vins kitti_odom_test [config file] [data folder] \n"
			   "for example: rosrun vins kitti_odom_test "
			   "~/catkin_ws/src/VINS-Fusion/config/kitti_odom/kitti_config00-02.yaml "
			   "/media/tony-ws1/disk_D/kitti/odometry/sequences/00/ \n");
		return 1;
	}

	string config_file = argv[1];
	printf("config_file: %s\n", argv[1]);
        
        //将参数从config中读取设置，主要包括是否是双目（是），是否使用IMU（否），和相机外参矫正参数，并且是单线程
	readParameters(config_file);
	estimator.setParameter();
	registerPub(n);
        ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 1100, img0_callback);
        ros::Subscriber sub_img1 = n.subscribe(IMAGE1_TOPIC, 1100, img1_callback);
//        message_filters::Subscriber<sensor_msgs::Image> image_LEFT(n, IMAGE0_TOPIC,2000);
//        message_filters::Subscriber<sensor_msgs::Image> image_RIGHT(n, IMAGE1_TOPIC, 2000);
//        message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(image_LEFT, image_RIGHT, 2000);
 //       sync.registerCallback(boost::bind(&callback, _1, _2));
        //depthCloud是将点云信息的坐标系转换到当前图像帧平面坐标系，但是还没有进行投影
      //  ros::Subscriber subdepthMap = n.subscribe("/depth_cloud",1100,depthCloud_callback);
	
	//处理接收点云的信息
	ros::Subscriber subCornerPointsSharp = n.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1100, laserCloudSharpHandler);

	ros::Subscriber subCornerPointsLessSharp = n.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1100, laserCloudLessSharpHandler);

	ros::Subscriber subSurfPointsFlat = n.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1100, laserCloudFlatHandler);

	ros::Subscriber subSurfPointsLessFlat = n.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1100, laserCloudLessFlatHandler);

	ros::Subscriber subLaserCloudFullRes = n.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 1100, laserCloudFullResHandler);

        std::thread sync_thread{sync_process};


        ros::spin();
	return 0;
}
