#include "Expectation.h"
#include <Windows.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>


void setup_glut_callbacks()
{
    glutInitDisplayMode(GLUT_DOUBLE);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);

    float vertices[] = {
        -1, -1, 0,
        1, -1, 0,
        1, 1, 0,
        -1, 1, 0,
    }; 

    static GLuint vertex_buffer;
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*4, vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0);

    glEnableVertexAttribArray(0);

    static GLuint first_frame = glutGet(GLUT_ELAPSED_TIME);
    static GLuint next_report = first_frame + 1000;
    static int frame_count = 0;

    glDepthFunc(GL_LEQUAL);

    static float roi_x = 0, roi_y = 0, roi_width = 1, roi_height = 1;

    glutKeyboardFunc([](unsigned char key, int, int)
    {
        switch(key)
        {
        case ' ':
            first_frame = glutGet(GLUT_ELAPSED_TIME);
            frame_count = 0;
            break;

        case 'w': roi_y += roi_height/4; break;
        case 's': roi_y -= roi_height/4; break;
        case 'a': roi_x -= roi_width/4; break;
        case 'd': roi_x += roi_width/4; break;

        case '1': roi_width /= 1.5; roi_height /= 1.5; break;
        case '2': roi_width *= 1.5; roi_height *= 1.5; break;

        case 27: //ESC
        case 'q':
        case 'Q':
            exit(EXIT_SUCCESS);
        }
    });

    glutDisplayFunc([]()
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glUniform2f(1, roi_x, roi_y);
        glUniform2f(0, roi_width, roi_height);

        glBindVertexArray(vertex_buffer);
        glDrawArrays(GL_QUADS, 0, 4);

        GLuint current_time = glutGet(GLUT_ELAPSED_TIME);
        ++frame_count;
        if (next_report < current_time)
        {
            std::stringstream ss;
            ss << "FPS: " << 1000.0*frame_count/(current_time-first_frame) << " (" << (1 / roi_width) << ")";
            glutSetWindowTitle(ss.str().c_str());
            next_report = current_time + 1000;
        }

        glutSwapBuffers();
        glutPostRedisplay();
    });

    glutReshapeFunc([](GLsizei width, GLsizei height)
    {
        if (height == 0) height = 1; 
        GLfloat aspect = (GLfloat)width / (GLfloat)height;
 
        glViewport(0, 0, width, height);
 
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0f, aspect, 0.1f, 100.0f);
    });
}

GLuint create_shader(GLenum type, std::string code)
{
    GLuint shader_id = glCreateShader(type);
    char const * const code_ptr[] = {code.c_str()};
    glShaderSource(shader_id, 1, code_ptr, nullptr);
    glCompileShader(shader_id);

    GLint compile_status;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compile_status);

    if(!compile_status)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader_id, 512, NULL, infoLog);
        glDeleteShader(shader_id);
        std::cout << "Shader compilation failed\n" << infoLog << std::endl;
        throw std::runtime_error("");
    }

    return shader_id;
}

void link_program(GLuint program)
{
    glLinkProgram(program);

    GLint link_status;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);

    if(!link_status)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "Program linking failed\n" << infoLog << std::endl;
        throw std::runtime_error("");
    }
}

int main(int argc, char *argv[]) try
{
    glutInit(&argc, argv);
    glutCreateWindow("GLEW Window"); // returns [int] Window ID

    std::cout << "OpenGL Version = " << glGetString(GL_VERSION) << std::endl;

    glewInit()  |  equals(GLEW_OK, ERROR_MSG("glewInit()"));

    GLuint vertex_shader = create_shader(GL_VERTEX_SHADER, 
            "#version 430 core\n"

            "layout (location = 0) in vec3 vertex_position;\n"
            "out vec2 position;\n"

            "void main()\n"
            "{\n"
            "   gl_Position = vec4(vertex_position, 1.0);\n"
            "   position = vec2(vertex_position);\n"
            "}\n");

    GLuint fragment_shader = create_shader(GL_FRAGMENT_SHADER, 
            "#version 430 core\n"

            "out vec4 FragColor;\n"
            "in vec2 position;\n"

            "layout(location = 1) uniform vec2 roi_pos;\n"
            "layout(location = 0) uniform vec2 roi_size;\n"

            "int f(vec2 p)\n"
            "{\n"
            "   dvec2 z = dvec2(0, 0);\n"
            "   for (int i = 1; i < 1000; ++i)\n"
            "   {\n"
            "       z = dvec2(z.x*z.x - z.y*z.y, 2*z.x*z.y) + vec2(p);"
            "       if (z.x*z.x + z.y*z.y > 2)\n"
            "           return i;\n"
            "   }\n"
            "   return -1;\n"
            "}\n"

            "void main()\n"
            "{\n"
            "   FragColor = vec4(0, 0, 0, 1);\n"
            "   int iter = f(vec2(position.x*roi_size.x, position.y*roi_size.y) + roi_pos);\n"
            "   if (iter == -1)\n"
            "      FragColor = vec4(0, 0, 0, 1);\n"
            "   else\n"
            "      FragColor = vec4(vec3(1, 1, 1)*(1-iter/1000.f), 1);\n"
            "}\n");

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    link_program(program);

    glUseProgram(program);

    glGetUniformLocation(program, "roi_pos")  |  equals(1, ERROR_MSG(""));
    glGetUniformLocation(program, "roi_size")  |  equals(0, ERROR_MSG(""));

    setup_glut_callbacks();
    glutMainLoop();
    return EXIT_SUCCESS;
}
catch(std::exception const &ex)
{
    std::cerr << "Failed with error: <" << typeid(ex).name() << ">: " << ex.what() << std::endl;
    return EXIT_FAILURE;
}