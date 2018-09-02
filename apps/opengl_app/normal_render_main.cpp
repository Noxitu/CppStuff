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
    glDepthFunc(GL_LEQUAL);

    glutKeyboardFunc([](unsigned char key, int, int)
    {
        switch(key)
        {
        case 27: //ESC
        case 'q':
        case 'Q':
            exit(EXIT_SUCCESS);
        }
    });

    float vertices[] = {
        0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,   // bottom right
        -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,   // bottom left
        0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f    // top 
    }; 

    static GLuint vertex_buffer;
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*6*3, vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3* sizeof(float)));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glutDisplayFunc([]()
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glBindVertexArray(vertex_buffer);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glutSwapBuffers();
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
            "#version 330 core\n"
            "layout (location = 0) in vec3 vertex_position;\n"
            "layout (location = 1) in vec4 input_vertex_color;\n"
            "out vec4 vertex_color;\n"
            "void main()\n"
            "{\n"
            "    gl_Position = vec4(vertex_position, 1.0);\n"
            "    vertex_color = input_vertex_color;\n"
            "}\n");

    GLuint fragment_shader = create_shader(GL_FRAGMENT_SHADER, 
            "#version 330 core\n"
            "out vec4 FragColor;\n"
            "in vec4 vertex_color;\n"

            "void main()\n"
            "{\n"
            "    FragColor = vertex_color;\n"
            "}\n");

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    link_program(program);

    glUseProgram(program);

    setup_glut_callbacks();
    glutMainLoop();
    return EXIT_SUCCESS;
}
catch(std::exception const &ex)
{
    std::cerr << "Failed with error: <" << typeid(ex).name() << ">: " << ex.what() << std::endl;
    return EXIT_FAILURE;
}