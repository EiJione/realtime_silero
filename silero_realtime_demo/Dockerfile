# Use the GCC 12.2.0 Bullseye image as the base image
FROM gcc:12.2.0-bullseye

# Install necessary packages

COPY data  /root/data/
COPY silero_vad.onnx /root/
COPY onnxruntime-linux-x64-1.12.1 /root/onnxruntime-linux-x64-1.12.1
# Copy your C++ source file into the Docker image
COPY realtime_silero.cpp /root/
COPY wav.h /root/
# Set working directory
WORKDIR /root/

# Command to build your C++ application
RUN g++ realtime_silero.cpp -I /root/onnxruntime-linux-x64-1.12.1/include/ -L /root/onnxruntime-linux-x64-1.12.1/lib/ -lonnxruntime -Wl,-rpath,/root/onnxruntime-linux-x64-1.12.1/lib/ -o test
# Command to run when starting the container
ENTRYPOINT ["./test"]


