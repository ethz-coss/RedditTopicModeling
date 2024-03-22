FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ENV TZ=Europe/Brussels

RUN apt-get update --fix-missing && apt-get install -y \
   build-essential \
   python3 \
   python3-dev \
   python3-pip

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0



# Install the package
# RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config
# RUN add-apt-repository universe && apt update && apt install -y software-properties-common python3 python3-pip python3-dev 
# RUN apt update && apt install -y ninja-build build-essential pkg-config
# RUN python -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama_cpp_python --verbose
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=ON" pip install llama_cpp_python==0.2.55 --verbose
RUN pip install anyio starlette fastapi uvicorn sse-starlette pydantic-settings starlette-context

#COPY . .
# Run the server
CMD python3 -m llama_cpp.server --n_gpu_layers 40 