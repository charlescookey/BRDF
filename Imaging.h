#pragma once


#include "GamesEngineeringBase.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define __STDC_LIB_EXT1__
#include "stb_image_write.h"

#include <vector>

// Stop warnings about buffer overruns if size is zero. Size should never be zero and if it is the code handles it.
#pragma warning( disable : 6386)


void savePNG(std::string filename, GamesEngineeringBase::Window* canvas)
{
    stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3);
}

void saveHDR(std::string filename, Colour* hdrpixels, GamesEngineeringBase::Window* canvas)
{
    stbi_write_hdr(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, (float*)hdrpixels);
    delete[] hdrpixels;
}

void savePNG2(std::string filename, std::vector<Colour>& hdrpixels, GamesEngineeringBase::Window* canvas)
{
    int width = canvas->getWidth();
    int height = canvas->getHeight();

    // Convert float to unsigned char
    unsigned char* pngData = new unsigned char[width * height * 3];
    for (int i = 0; i < width * height; i++) {
        pngData[i * 3 + 0] = (unsigned char)(hdrpixels[i].r * 255);
        pngData[i * 3 + 1] = (unsigned char)(hdrpixels[i].g * 255);
        pngData[i * 3 + 2] = (unsigned char)(hdrpixels[i].b * 255);
    }

    stbi_write_png(filename.c_str(), width, height, 3, pngData, width * 3);

    delete[] pngData;
}