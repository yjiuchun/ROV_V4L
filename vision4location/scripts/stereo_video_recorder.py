#!/usr/bin/env python3

"""
ROS node that subscribes to stereo image topics (/left and /right by default)
and records each stream into a separate video file using time-synchronised frames.
"""

import threading
from datetime import datetime
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber, TimeSynchronizer
from sensor_msgs.msg import Image


class StereoVideoRecorder:
    """Synchronises left/right image topics and writes them to disk as videos."""

    def __init__(self):
        self.bridge = CvBridge()
        self.left_writer = None
        self.right_writer = None
        self.writer_lock = threading.Lock()

        # self.left_topic = rospy.get_param("~left_topic", "/zed2i/zed_node/left/image_rect_color")
        # self.right_topic = rospy.get_param("~right_topic", "/zed2i/zed_node/right/image_rect_color")

        self.left_topic = rospy.get_param("~left_topic", "/zed2i/zed_node/left_raw/image_raw_color")
        self.right_topic = rospy.get_param("~right_topic", "/zed2i/zed_node/right_raw/image_raw_color")


        self.output_dir = Path(
            rospy.get_param("~output_dir", str(Path("~").expanduser() / "stereo_videos"))
        ).expanduser().resolve()
        self.file_basename = rospy.get_param("~file_basename", "stereo_capture")
        self.file_extension = rospy.get_param("~file_extension", "mp4").lstrip(".")
        self.codec = rospy.get_param("~codec", "mp4v")
        self.fps = rospy.get_param("~fps", 30.0)
        self.cv_encoding = rospy.get_param("~cv_encoding", "bgr8")
        queue_size = rospy.get_param("~sync_queue_size", 10)
        sync_slop = rospy.get_param("~sync_slop", 0.02)
        self.use_exact_sync = rospy.get_param("~exact_sync", False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.left_path = self.output_dir / f"{self.file_basename}_left_{timestamp}.{self.file_extension}"
        self.right_path = self.output_dir / f"{self.file_basename}_right_{timestamp}.{self.file_extension}"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.left_sub = Subscriber(self.left_topic, Image)
        self.right_sub = Subscriber(self.right_topic, Image)

        if self.use_exact_sync:
            rospy.loginfo("StereoVideoRecorder using exact time synchronizer.")
            synchronizer = TimeSynchronizer([self.left_sub, self.right_sub], queue_size)
        else:
            rospy.loginfo("StereoVideoRecorder using approximate time synchronizer (slop %.3fs).", sync_slop)
            synchronizer = ApproximateTimeSynchronizer(
                [self.left_sub, self.right_sub],
                queue_size=queue_size,
                slop=sync_slop,
                allow_headerless=False,
            )

        synchronizer.registerCallback(self.synced_callback)
        rospy.on_shutdown(self._release_writers)

        rospy.loginfo(
            "StereoVideoRecorder initialised.\n Left topic: %s\n Right topic: %s\n Output directory: %s",
            self.left_topic,
            self.right_topic,
            str(self.output_dir),
        )

    def synced_callback(self, left_msg: Image, right_msg: Image):
        """Handle synchronised image pairs and append them to their respective videos."""
        try:
            left_frame = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding=self.cv_encoding)
            right_frame = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding=self.cv_encoding)
        except CvBridgeError as exc:
            rospy.logerr("CvBridge conversion failed: %s", exc)
            return

        with self.writer_lock:
            if self.left_writer is None:
                self.left_writer = self._create_writer(self.left_path, left_frame)
            if self.right_writer is None:
                self.right_writer = self._create_writer(self.right_path, right_frame)

            if self.left_writer is None or self.right_writer is None:
                rospy.logerr_throttle(5.0, "VideoWriter was not initialised correctly; dropping frames.")
                return

            self.left_writer.write(left_frame)
            self.right_writer.write(right_frame)

    def _create_writer(self, file_path: Path, sample_frame):
        """Create a cv2.VideoWriter for the provided sample frame."""
        height, width = sample_frame.shape[:2]
        is_color = sample_frame.ndim == 3 and sample_frame.shape[2] >= 3
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(file_path), fourcc, self.fps, (width, height), isColor=is_color)

        if not writer.isOpened():
            rospy.logerr("Failed to open VideoWriter for %s", file_path)
            return None

        rospy.loginfo("Recording video to %s (%dx%d @ %.2f fps).", file_path, width, height, self.fps)
        return writer

    def _release_writers(self):
        """Flush and close both video writers."""
        with self.writer_lock:
            for writer, label in ((self.left_writer, "left"), (self.right_writer, "right")):
                if writer is not None:
                    writer.release()
                    rospy.loginfo("Closed %s video file.", label)
            self.left_writer = None
            self.right_writer = None


def main():
    rospy.init_node("stereo_video_recorder")
    StereoVideoRecorder()
    rospy.loginfo("StereoVideoRecorder node started. Waiting for images...")
    rospy.spin()


if __name__ == "__main__":
    main()

