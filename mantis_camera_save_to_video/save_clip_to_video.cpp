/******************************************************************************
 *
 * GetClipMcamImages.c
 * Author: Andrew Ferg
 *
 * This example app uses the Mantis API to retrieve all the images for
 * a specified microcamera ID between a given start and end time and
 * save them to a specified storage directory with associated JSON
 * metadata files.
 *
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include "mantis/MantisAPI.h"

/**
 * \brief Function that handles new ACOS_CAMERA objects
 **/
void newCameraCallback(ACOS_CAMERA cam, void* data)
{
    static int cameraCounter = 0;
    ACOS_CAMERA* camList = (ACOS_CAMERA*) data;
    camList[cameraCounter++] = cam;
}

/**
 * \brief prints the command line options
 **/
void printHelp()
{
   printf("GetClipMcamImages Demo Application\n");
   printf("Usage:\n");
   printf("\t-t <start> <end> The start and end times of the clip\n");
   printf("\t-f <framerate> The framerate the clip was captured at\n");
   printf("\t-h Prints this help message and exits\n");
   printf("\t-ip <address> IP Address connect to (default localhost)\n");
   printf("\t-port <port> Port connect to (default 9999)\n");
   printf("\t-mcam <mcam ID> The ID of the microcamera to get images for (default behavior gets all microcameras for the clip\n");
   printf("\t-dir <directory> The directory to save the JPEGs to (default .)\n");
}

/**
 * \brief Main function
 **/
int main(int argc, char * argv[])
{
    /* Parse command line inputs to determine IP address
     * or port if provided from the command line */
    char ip[24] = "localhost";
    int port = 9999;
    char dir[256] = ".";
    uint64_t startTime = 0;
    uint64_t endTime = 0;
    double framerate = 0;
    uint32_t mcamID = 0;
    for( int i = 1; i < argc; i++ ){
        if( !strcmp(argv[i],"-ip") ){
            if( ++i >= argc ){
                printHelp();
                return 0;
            }
            int length = strlen(argv[i]);
            if( length < 24 ){
                strncpy(ip, argv[i], length);
                ip[length] = 0;
            }
        } else if( !strcmp(argv[i],"-port") ){
            if( ++i >= argc ){
                printHelp();
                return 0;
            }
            int length = strlen(argv[i]);
            port = atoi(argv[i]);
        } else if( !strcmp(argv[i],"-t") ){
            if( ++i >= argc ){
                printHelp();
                return 0;
            }
            startTime = strtoul(argv[i], NULL, 10);
            if( ++i >= argc ){
                printHelp();
                return 0;
            }
            endTime = strtoul(argv[i], NULL, 10);
        } else if( !strcmp(argv[i],"-mcam") ){
            if( ++i >= argc ){
                printHelp();
                return 0;
            }
            mcamID = strtol(argv[i], NULL, 10);
        } else if( !strcmp(argv[i],"-f") ){
            if( ++i >= argc ){
                printHelp();
                return 0;
            }
            framerate = atof(argv[i]);
        } else if( !strcmp(argv[i],"-dir") ){
            if( ++i >= argc ){
                printHelp();
                return 0;
            }
            strcpy(dir, argv[i]);
        } else if( !strcmp(argv[i], "-h") ){
            printHelp();
            return 1;
        } else{
            printHelp();
            return 0;
        }
    }

    if( framerate == 0 || startTime == 0 || endTime == 0 ){
        printf("Start time, end time, and framerate are required arguments\n");
        printHelp();
        return 0;
    }

    /* connect to the V2 instance */
    connectToCameraServer(ip, port);
    sleep(1);

    /* get cameras from API */
    int numCameras = getNumberOfCameras();
    ACOS_CAMERA cameraList[numCameras];
    NEW_CAMERA_CALLBACK camCB;
    camCB.f = newCameraCallback;
    camCB.data = cameraList;
    setNewCameraCallback(camCB);
    printf("API connected to %d Mantis systems\n", numCameras);

    /****************************************************************
     * THE REST OF THIS EXAMPLE WILL USE THE FIRST CAMERA IN THE LIST 
     ****************************************************************/
    ACOS_CAMERA myMantis = cameraList[0];

    /* If the camera struct reports 0 microcameras, then it has never been
     * connected before and we must establish a connection to retrieve the
     * correct number of microcameras */
    if( myMantis.numMCams == 0 ){
        if( !toggleConnection(myMantis, true, 5000) ){
            printf("Failed to establish connection for camera %u!\n",
                   myMantis.camID);
            return 0;
        } else{
            printf("Camera %u is now connected to its physical camera system\n",
                   myMantis.camID);
            sleep(1);
        }
        myMantis.numMCams = getCameraNumberOfMCams(myMantis);
    }

    /* Next, get the microcameras for the Mantis so we know what to request.
     * Note: the ACOS_CAMERA struct in the returned ACOS_CLIP struct should be 
     * identical to the one used in the start/stop recording commands
     * unless the struct was corrupted by unsafe use of the API */
    MICRO_CAMERA mcamList[myMantis.numMCams];
    getCameraMCamList(myMantis, mcamList, myMantis.numMCams);

    for( int i = 0; i < myMantis.numMCams; i++ ){
        printf("found mcam with ID: %u\n", mcamList[i].mcamID);
    }

    /* if a specific mcam was chosen, remove the rest form the list */
    // int numMCams = (mcamID == 0) ? myMantis.numMCams : 1;
    // printf("Requesting frames for %d microcameras\n", numMCams);
    // if( mcamID != 0 ){
    //     MICRO_CAMERA mcam;
    //     for( int i = 0; i < myMantis.numMCams; i++ ){
    //         if( mcamList[i].mcamID == mcamID ){
    //             mcam = mcamList[i];
    //         } else{
    //             MICRO_CAMERA mc;
    //             mcamList[i] = mc;
    //         }
    //     }
    //     mcamList[0] = mcam;
    // }

    /* Next we calculate the length of a frame in microseconds */
    uint64_t frameLength = (uint64_t)(1.0/framerate * 1e6);

    /* Now for each microcamera, we request frames starting at the 
     * startTime and increment the time of our requests by the length 
     * of a frame until we reach the endTime, Unlike when requesting
     * the most recent frame, requesting a specific time may fail
     * if a frame was dropped, so it is good to check that the image
     * buffer pointer is not NULL before interacting with the frame */
    uint64_t requestCounter = 0;
    uint64_t frameCounter = 0;

    for (int i = 0; i < myMantis.numMCams; i ++) {
        cv::VideoWriter outputVideo;
        char videoName[512];
        sprintf(videoName, "%s/%u.avi",
                dir,
                mcamList[i].mcamID);
        outputVideo.open(videoName, CV_FOURCC('D','I','V','3'), 30, cv::Size(3840, 2160), true);
        if (!outputVideo.isOpened()) {
            printf("Failed to open file to write!\n");
            exit(0);
        }
        char configName[512];
        sprintf(configName, "%s/%u.txt",
                dir,
                mcamList[i].mcamID);
        FILE *fp;
        fp = fopen(configName, "w");

        int frameInd = 0;

        for( uint64_t t = startTime; t < endTime; t += frameLength ){
            /* get the next frame for this mcam */
            printf("%d: Requesting frame for mcam %u\n", frameInd, mcamList[i].mcamID);
            requestCounter++;
            FRAME frame = getFrame(myMantis, 
                                   mcamList[i].mcamID,
                                   t,
                                   ATL_TILING_1_1_2,
                                   ATL_TILE_4K);
            /* check that the request succeeded before using the frame */
            if( frame.m_image != NULL ){
                frameCounter++;
                char fileName[512];
                sprintf(fileName, "%s/%u_%lu.jpg",
                        dir,
                        frame.m_metadata.m_camId,
                        frame.m_metadata.m_timestamp);
                printf("Saving image %s, mcam:%u, timestamp: %lu\n", 
                       fileName, 
                       frame.m_metadata.m_camId, 
                       frame.m_metadata.m_timestamp);
                // if (frameInd == 0)
                //     saveMCamFrame(frame, fileName);

                fprintf(fp, "%d\t%u\t%lu\n", frameInd, frame.m_metadata.m_camId, frame.m_metadata.m_timestamp);

                // printf("Image size: %u\n", frame.m_metadata.m_size);
                cv::Mat rawdata(1, frame.m_metadata.m_size, CV_8UC1, (uchar*)frame.m_image);
                cv::Mat img = cv::imdecode(rawdata, 1);
                // if (frameInd == 0)
                //     cv::imwrite(fileName, img);
                
                outputVideo << img;

                // printf("Image information: row: %d, col: %d, filename: %s\n", img.rows, img.cols, fileName);
                // if (frameInd < 3)
                //     cv::imwrite(fileName, img);
                
                
                /* return the frame buffer pointer to prevent memory leaks */
                if( !returnPointer(frame.m_image) ){
                    printf("Failed to return the pointer for the frame buffer\n");
                }
            } else{
                fprintf(fp, "%d\t%u\t%lu\n", frameInd, 0, 0);
                printf("Frame request failed!\n");
            }

            frameInd ++;

        }
        fclose(fp);
        outputVideo.release();

        printf("Received %lu of %lu requested frames across %d microcameras\n",
            frameCounter,
            requestCounter,
            myMantis.numMCams);
    }
    /* Disconnect the cameras to prevent issues when another program 
     * tries to connect */
    for( int i = 0; i < numCameras; i++ ){
        disconnectCamera(cameraList[i]);
    }

    exit(1);
}