# waifu2x ncnn Vulkan Python

![CI](https://github.com/ArchieMeng/waifu2x-ncnn-vulkan-python/workflows/CI/badge.svg)

## Introduction

[waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan) is nihui's ncnn implementation of waifu2x converter. Runs fast on Intel / AMD / Nvidia with Vulkan API.

This project is a Python wrapper of nihui's project.

waifu2x-ncnn-vulkan-python wraps [waifu2x-ncnn-vulkan project](https://github.com/nihui/waifu2x-ncnn-vulkan) by SWIG to make it easier to integrate waifu2x-ncnn-vulkan with existing python projects.

This project only wrapped the original Waifu2x class. As a result, functions other than the core upscaling and denoising such as multi-thread loading and saving are not available. Of course, the auto tilesize and prepadding settings are implements, so don't worry about them.

## Download

linux x64, Windows x64 and MacOS x64 releases are available now. For other platforms, you may compile it on your own.
The reason why MacOS ARM64 build is not available is that it needs ARM Python Dev Libs which I have no ideas on how to
get it on Github's MacOS x64 VM. Moreover, I don't have a Mac.
For Linux distro with GLIBC < 2.29 (like Ubuntu 18.04), ubuntu-1804 pre-built should be used.


## Build

First, you have to install python, python development package (Python native development libs in Visual Studio), vulkan SDK and SWIG on your platform. And then:

### Linux
1. install dependencies: cmake, vulkan sdk, swig and python-dev

Debian, Ubuntu and other Debian-like Distros
```shell
apt-get install cmake libvulkan-dev swig python3-dev
```
Arch Distros
```shell
pacman -S base-devel cmake vulkan-headers vulkan-icd-loader swig python
````

2. Build with CMake
```shell
git clone https://github.com/ArchieMeng/waifu2x-ncnn-vulkan-python.git
cd waifu2x-ncnn-vulkan-python
git submodule update --init --recursive
cd src
cmake -B build .
cd build
make
```

### Windows
I used Visual Studio 2019 and msvc v142 to build this project for Windows.

Install visual studio and open the project directory, and build. Job done.

The only problem on Windows is that, you cannot use [CMake for Windows](https://cmake.org/download/) to generate the Visual Studio solution file and build it. This will make the lib crash on loading.

The only way is [use Visual Studio to open the project as directory](https://www.microfocus.com/documentation/visual-cobol/vc50/VS2019/GUID-BE1C48AA-DB22-4F38-9644-E9B48658EF36.html), and build it from Visual Studio.

1. install dependencies: cmake, vulkan sdk, swig and python-dev
- download vulkan sdk from https://vulkan.lunarg.com/sdk/home
- download SWIG from http://www.swig.org/download.html
- install python with dev libs either from official website or Visual Studio 
  (The python dev environment and python native dev libs components)
2. cmake the project. Either reference the linux build method or just do it on Visual Studio.

### Mac OS X
1. install dependencies: cmake, vulkan sdk, swig and python-dev
- download vulkan sdk from https://vulkan.lunarg.com/sdk/home
- If you have homebrew installed, run the command below to get SWIG
```shell
brew install swig
```
- I guess python dev is out-of-box in Mac. If not, google it.
    

2. Build with CMake
- You can pass -DUSE_STATIC_MOLTENVK=ON option to avoid linking the vulkan loader library on MacOS
```shell
git clone https://github.com/ArchieMeng/waifu2x-ncnn-vulkan-python.git
cd waifu2x-ncnn-vulkan-python
git submodule update --init --recursive
cd src
cmake -B build .
cd build
make
```
## Usages

### Example program

```python
from PIL import Image
from waifu2x_ncnn_vulkan import Waifu2x

im = Image.open("0.jpg")
w2x_obj = Waifu2x(gpuid=0)
out_im = w2x_obj.process(im)
out_im.save("1.png")
```

## [Docs](Docs.md)

## Known issues
- [Module finalization will crash for nvidia dedicated graphics card(s) on Linux. (The image processing still works.)](https://github.com/Tencent/ncnn/issues/2666)
- Not yet tested for Mac OS. I guess it should work.

## Original waifu2x Project

- https://github.com/nagadomi/waifu2x
- https://github.com/lltcggie/waifu2x-caffe
- https://github.com/nihui/waifu2x-ncnn-vulkan

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/webmproject/libwebp for encoding and decoding Webp images on ALL PLATFORMS
- https://github.com/nothings/stb for decoding and encoding image on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directory on Windows
