mkdir -p weights
wget -c "https://pjreddie.com/media/files/yolov3.weights" --header "Referer: pjreddie.com" -O "weights/yolov3.weights"
wget -c "https://pjreddie.com/media/files/darknet53.conv.74" --header "Referer: pjreddie.com" -O "weights/darknet53.conv.74"