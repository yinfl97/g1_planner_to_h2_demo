/**
 * Minimal GLFW declarations used by the standalone planner demo.
 *
 * sonic_sim ships GLFW as part of the Python glfw wheel but does not ship its
 * development header. Keeping this tiny ABI declaration avoids adding a
 * system package solely for the standalone MuJoCo viewer.
 */
#ifndef H2_PLANNER_TEST_GLFW_MINIMAL_HPP
#define H2_PLANNER_TEST_GLFW_MINIMAL_HPP

extern "C" {

typedef struct GLFWwindow GLFWwindow;
typedef struct GLFWmonitor GLFWmonitor;

int glfwInit(void);
void glfwTerminate(void);
GLFWwindow* glfwCreateWindow(int width, int height, const char* title,
                             GLFWmonitor* monitor, GLFWwindow* share);
void glfwDestroyWindow(GLFWwindow* window);
void glfwMakeContextCurrent(GLFWwindow* window);
void glfwSwapInterval(int interval);
int glfwWindowShouldClose(GLFWwindow* window);
void glfwSetWindowShouldClose(GLFWwindow* window, int value);
void glfwPollEvents(void);
void glfwSwapBuffers(GLFWwindow* window);
int glfwGetKey(GLFWwindow* window, int key);
void glfwGetFramebufferSize(GLFWwindow* window, int* width, int* height);
void glfwSetWindowTitle(GLFWwindow* window, const char* title);
double glfwGetTime(void);

}  // extern "C"

inline constexpr int GLFW_PRESS = 1;
inline constexpr int GLFW_KEY_SPACE = 32;
inline constexpr int GLFW_KEY_A = 65;
inline constexpr int GLFW_KEY_D = 68;
inline constexpr int GLFW_KEY_E = 69;
inline constexpr int GLFW_KEY_F = 70;
inline constexpr int GLFW_KEY_Q = 81;
inline constexpr int GLFW_KEY_R = 82;
inline constexpr int GLFW_KEY_S = 83;
inline constexpr int GLFW_KEY_W = 87;
inline constexpr int GLFW_KEY_ESCAPE = 256;
inline constexpr int GLFW_KEY_RIGHT = 262;
inline constexpr int GLFW_KEY_LEFT = 263;
inline constexpr int GLFW_KEY_DOWN = 264;
inline constexpr int GLFW_KEY_UP = 265;

#endif  // H2_PLANNER_TEST_GLFW_MINIMAL_HPP
