#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace cv;
using Clock = std::chrono::steady_clock;

static string readText(const string& path) {
    ifstream f(path, ios::in | ios::binary);
    if (!f) throw runtime_error("Cannot open file: " + path);
    return { istreambuf_iterator<char>(f), istreambuf_iterator<char>() };
}

static void glfwErrorCb(int code, const char* desc) {
    cerr << "[GLFW] (" << code << ") " << desc << endl;
}

static GLuint compile(GLenum type, const string& src) {
    GLuint sh = glCreateShader(type);
    const char* s = src.c_str();
    glShaderSource(sh, 1, &s, nullptr);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        string log(len, '\0'); glGetShaderInfoLog(sh, len, nullptr, log.data());
        throw runtime_error(string("Shader compile failed:\n") + log);
    }
    return sh;
}

static GLuint link(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        string log(len, '\0'); glGetProgramInfoLog(p, len, nullptr, log.data());
        throw runtime_error(string("Program link failed:\n") + log);
    }
    return p;
}

//FUNCTIONS FOR ASSIGMENT
// CPU Pixelate: downsample then upsample
static void cpuPixelate(cv::Mat& rgb, int block) {
    if (block < 2) return;
    cv::Mat small;
    cv::resize(rgb, small, cv::Size(rgb.cols / block, rgb.rows / block), 0, 0, cv::INTER_NEAREST);
    cv::resize(small, rgb, rgb.size(), 0, 0, cv::INTER_NEAREST);
}

// CPU Comic Art: edge + posterize
static void cpuComic(cv::Mat& rgb, double edgeThresh = 0.25) {
    // Convert to grayscale for edge detection
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.0);

    // Laplacian edge detection
    cv::Mat edges;
    cv::Laplacian(gray, edges, CV_32F, 3);

    // Convert edges to absolute magnitude and threshold
    cv::Mat mag;
    cv::convertScaleAbs(cv::abs(edges), mag);
    cv::Mat mask;
    cv::threshold(mag, mask, edgeThresh * 255.0, 255, cv::THRESH_BINARY);

    // --- Posterize colors (integer bands) ---
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0); // normalize 0–1
    cv::Mat tmp(rgb.size(), rgb.type());

    // Quantize to 5 color levels
    int levels = 5;
    rgb.forEach<cv::Vec3f>([&](cv::Vec3f& pix, const int*) {
        for (int i = 0; i < 3; ++i)
            pix[i] = std::floor(pix[i] * levels) / levels;
        });

    rgb.convertTo(rgb, CV_8UC3, 255.0);

    // --- Apply edges as black lines ---
    rgb.setTo(cv::Scalar(0, 0, 0), mask);
}

// ===========================================================
//  Affine transformation helpers (used by both CPU and GPU)
// ===========================================================

struct XformState {
    float tx = 0.0f, ty = 0.0f;
    float scale = 1.0f;
    float angle = 0.0f;
    double lastX = 0.0, lastY = 0.0;
    bool draggingPan = false;
    bool draggingRotate = false;
};

static void buildAffinePixels(const XformState& xf, int W, int H, float M[9]) {
    float cx = 0.5f * W, cy = 0.5f * H;
    float c = cosf(xf.angle), s = sinf(xf.angle);
    float invS = 1.0f / std::max(1e-6f, xf.scale);
    float ci = c, si = -s;

    float m[9] = { 1,0,0, 0,1,0, 0,0,1 };
    auto mul = [](const float A[9], const float B[9], float R[9]) {
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            R[3 * r + c] = A[3 * r + 0] * B[3 * 0 + c] + A[3 * r + 1] * B[3 * 1 + c] + A[3 * r + 2] * B[3 * 2 + c];
        };
    auto T = [&](float tx, float ty) { float t[9] = { 1,0,tx,0,1,ty,0,0,1 }; float r[9]; mul(t, m, r); memcpy(m, r, sizeof(r)); };
    auto R = [&](float cc, float ss) { float r_[9] = { cc,-ss,0,ss,cc,0,0,0,1 }; float r2[9]; mul(r_, m, r2); memcpy(m, r2, sizeof(r2)); };
    auto S = [&](float sx, float sy) { float s_[9] = { sx,0,0,0,sy,0,0,0,1 }; float r2[9]; mul(s_, m, r2); memcpy(m, r2, sizeof(r2)); };

    T(cx, cy); S(invS, invS); R(ci, si); T(-cx, -cy); T(-xf.tx, -xf.ty);
    memcpy(M, m, sizeof(m));
}

static void affinePixelsToUV(const float M[9], int W, int H, float N[9]) {
    auto mul3 = [](const float A[9], const float B[9], float R[9]) {
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            R[3 * r + c] = A[3 * r + 0] * B[3 * 0 + c] + A[3 * r + 1] * B[3 * 1 + c] + A[3 * r + 2] * B[3 * 2 + c];
        };
    float S1[9] = { 1.0f / W,0,0, 0,1.0f / H,0, 0,0,1 };
    float S2[9] = { float(W),0,0, 0,float(H),0, 0,0,1 };
    float tmp[9]; mul3(M, S2, tmp); mul3(S1, tmp, N);
}

static cv::Matx23f toCv2x3(const float M[9]) {
    return cv::Matx23f(M[0], M[1], M[2],
        M[3], M[4], M[5]);
}



int main(int argc, char** argv) try {
    // --- Open video (file if given, else webcam) ---
    VideoCapture cap;
    if (argc > 1) cap.open(argv[1]);
    else          cap.open(0, cv::CAP_ANY);

    if (!cap.isOpened()) {
        cerr << "ERROR: Could not open video source." << endl;
        return 1;
    }

    Mat frameBGR; cap >> frameBGR;
    if (frameBGR.empty()) {
        cerr << "ERROR: First frame empty." << endl;
        return 1;
    }

    // --- GLFW + context ---
    glfwSetErrorCallback(glfwErrorCb);
    if (!glfwInit()) { cerr << "ERROR: glfwInit failed." << endl; return 1; }
    GLFWwindow* win = glfwCreateWindow(800, 600, "MyProject", nullptr, nullptr);
    if (!win) { cerr << "ERROR: glfwCreateWindow failed." << endl; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);  // disable vsync


    // --- GLEW (must be after context) ---
    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    glGetError(); // clear benign GL error
    if (glewErr != GLEW_OK) {
        cerr << "ERROR: glewInit failed: "
            << reinterpret_cast<const char*>(glewGetErrorString(glewErr)) << endl;
        return 1;
    }

    // --- Fullscreen quad ---
    float v[] = {
        -1.f,  1.f, 0.f, 1.f,  -1.f, -1.f, 0.f, 0.f,   1.f, -1.f, 1.f, 0.f,
        -1.f,  1.f, 0.f, 1.f,   1.f, -1.f, 1.f, 0.f,   1.f,  1.f, 1.f, 1.f
    };
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // --- Load shaders (filter shader) ---
    string vs = readText("shaders/basic.vert");
    string fs = readText("shaders/filters.frag");
    GLuint p = link(compile(GL_VERTEX_SHADER, vs), compile(GL_FRAGMENT_SHADER, fs));
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "texture1"), 0);

    // --- Uniform locations ---
    GLint locMode = glGetUniformLocation(p, "uMode");
    GLint locTexSize = glGetUniformLocation(p, "uTexSize");
    GLint locBlock = glGetUniformLocation(p, "uBlock");
    GLint locEdge = glGetUniformLocation(p, "uEdgeThresh");
    GLint locAff = glGetUniformLocation(p, "uAffine");   // <<-- for GPU affine
    if (locAff == -1) { std::cerr << "ERROR: uAffine not active\n"; }

    // --- Filter & transform state ---
    enum class Filter { None = 0, Pixelate = 1, Comic = 2 };
    bool  useGPU = true;
    Filter filter = Filter::None;
    int   pixelBlock = 16;
    float edgeThresh = 0.25f;

    XformState xf; // persistent pan/zoom/rotate
    xf.scale = 0.7f;

    // --- Texture ---
    Mat frameRGB; cvtColor(frameBGR, frameRGB, COLOR_BGR2RGB);
    GLuint tex = 0; glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frameRGB.cols, frameRGB.rows, 0,
        GL_RGB, GL_UNSIGNED_BYTE, frameRGB.data);
    auto tStart = Clock::now();
    auto tLastPrint = tStart;
    int frameCount = 0;


    // --- Main loop ---
    while (!glfwWindowShouldClose(win)) {
        cap >> frameBGR;
        if (frameBGR.empty()) break;
        cvtColor(frameBGR, frameRGB, COLOR_BGR2RGB);
        cv::flip(frameRGB, frameRGB, 0); // keep upright display

        // --- Input handling for transform (pan/zoom/rotate) ---
        double mx, my; glfwGetCursorPos(win, &mx, &my);
        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (!xf.draggingPan) { xf.draggingPan = true; xf.lastX = mx; xf.lastY = my; }
        }
        else xf.draggingPan = false;

        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!xf.draggingRotate) { xf.draggingRotate = true; xf.lastX = mx; xf.lastY = my; }
        }
        else xf.draggingRotate = false;

        if (xf.draggingPan) {
            float dx = float(mx - xf.lastX);
            float dy = float(my - xf.lastY);
            int winW, winH; glfwGetWindowSize(win, &winW, &winH);
            float sx = (float)frameRGB.cols / (float)winW;
            float sy = (float)frameRGB.rows / (float)winH;
            xf.tx += dx * sx;
            xf.ty -= dy * sy;
            xf.lastX = mx; xf.lastY = my;
        }
        if (xf.draggingRotate) {
            float dx = float(mx - xf.lastX);
            xf.angle += dx * 0.01f;
            xf.lastX = mx; xf.lastY = my;
        }
        // Zoom with +/- keys (simple)
        if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS) { xf.scale *= 1.05f; }
        if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS) { xf.scale /= 1.05f; }
        xf.scale = std::clamp(xf.scale, 0.1f, 10.0f);
        if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS) { xf = XformState{}; }

        // --- Build affine (dest->src) matrix once per frame ---
        float Mpix[9]; buildAffinePixels(xf, frameRGB.cols, frameRGB.rows, Mpix);
        float Muv[9]; affinePixelsToUV(Mpix, frameRGB.cols, frameRGB.rows, Muv);

        // --- CPU path: apply warpAffine, then CPU filters (shader passthrough) ---
        if (!useGPU) {
            cv::Mat warped;
            cv::warpAffine(frameRGB, warped, toCv2x3(Mpix), frameRGB.size(),
                cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            frameRGB = warped;

            if (filter == Filter::Pixelate)       cpuPixelate(frameRGB, pixelBlock);
            else if (filter == Filter::Comic)     cpuComic(frameRGB, edgeThresh);
        }

        // --- Upload frame ---
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frameRGB.cols, frameRGB.rows,
            GL_RGB, GL_UNSIGNED_BYTE, frameRGB.data);

        // --- Draw ---
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(p);

        // GPU uniforms
        glUniform2f(locTexSize, (float)frameRGB.cols, (float)frameRGB.rows);
        glUniform1f(locBlock, (float)pixelBlock);
        glUniform1f(locEdge, edgeThresh);
        glUniform1i(glGetUniformLocation(p, "texture1"), 0);
        glUniform1i(locMode, useGPU ? (int)filter : 0);

         // Send as row-major, ask GL to transpose on upload
        glUseProgram(p);
        glUniformMatrix3fv(locAff, 1, GL_TRUE, Muv);


        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glfwSwapBuffers(win);
        glfwPollEvents();

        // --- Filter/backend hotkeys ---
        static double last = 0.0;
        auto debounce = [&](int key, auto fn) {
            if (glfwGetKey(win, key) == GLFW_PRESS) {
                double now = glfwGetTime();
                if (now - last > 0.2) { fn(); last = now; }
            }
            };
        debounce(GLFW_KEY_F, [&] {
            filter = (filter == Filter::Comic) ? Filter::None
                : (Filter)((int)filter + 1);
            cout << "Filter: " << (int)filter << endl;
            });
        debounce(GLFW_KEY_G, [&] {
            useGPU = !useGPU;
            cout << (useGPU ? "GPU" : "CPU") << " mode\n";
            });
        debounce(GLFW_KEY_1, [&] { pixelBlock = max(2, pixelBlock - 1); });
        debounce(GLFW_KEY_2, [&] { pixelBlock = min(128, pixelBlock + 1); });
        debounce(GLFW_KEY_9, [&] { edgeThresh = max(0.05f, edgeThresh - 0.02f); });
        debounce(GLFW_KEY_0, [&] { edgeThresh = min(1.0f, edgeThresh + 0.02f); });

        frameCount++;
        auto now = Clock::now();
        double elapsed = std::chrono::duration<double>(now - tStart).count();
        double fps = frameCount / elapsed;

        // Print every 0.5 seconds
        double elapsedPrint = std::chrono::duration<double>(now - tLastPrint).count();
        if (elapsedPrint >= 0.5) {
            std::cout << "FPS: " << fps;

            // Optional: also print whether CPU or GPU is active
            std::cout << " | Mode: " << (useGPU ? "GPU" : "CPU");

            // Optional: print current filter
            std::string f =
                (filter == Filter::None) ? "None" :
                (filter == Filter::Pixelate) ? "Pixelate" : "Comic";
            std::cout << " | Filter: " << f << std::endl;

            tLastPrint = now;
        }

    }

    glDeleteTextures(1, &tex);
    glDeleteProgram(p);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glfwTerminate();
    return 0;

}
catch (const std::exception& e) {
    cerr << "FATAL: " << e.what() << endl;
    return 1;
}


