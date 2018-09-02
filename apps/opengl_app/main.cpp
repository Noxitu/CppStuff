#include "Expectation.h"
#include <Windows.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>

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
        std::cout << "Program linking failed:\n" << infoLog << std::endl;
        throw std::runtime_error("");
    }
}

int main(int argc, char *argv[]) try
{
    glutInit(&argc, argv);
    glutCreateWindow("GLEW Window"); // returns [int] Window ID

    std::cout << "OpenGL Version = " << glGetString(GL_VERSION) << std::endl;

    glewInit()  |  equals(GLEW_OK, ERROR_MSG("glewInit()"));

    GLuint compute_shader = create_shader(GL_COMPUTE_SHADER, 
            "#version 430 core\n"

            "layout( binding=4 ) buffer Buffer\n"
            "{\n"
            "   float values[];\n"
            "};\n"

            "layout( binding=5 ) buffer BufferInput\n"
            "{\n"
            "   float input_buffer[];\n"
            "};\n"

            "layout( binding=6 ) buffer BufferWeights\n"
            "{\n"
            "   float weights[];\n"
            "};\n"

            "layout( local_size_x = 1000, local_size_y = 1, local_size_z = 1 ) in;\n"

            "void main()\n"
            "{\n"
            "   float result = 0;\n"
            "   for (int i = 0; i < 300000; ++i)\n"
            "       result += input_buffer[gl_GlobalInvocationID.x] * weights[i];\n"
            "   values[gl_GlobalInvocationID.x] = result;\n"
            "}\n");

    const int N = 300000;

    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer( GL_SHADER_STORAGE_BUFFER, buffer );
    glBufferData( GL_SHADER_STORAGE_BUFFER, sizeof(float) * N, NULL, GL_STATIC_DRAW );

    glGetError()  |  equals(GL_NO_ERROR, ERROR_MSG("An error!"));

    {
        float *data = (float*) glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, N * sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        for (int i = 0; i < N; ++i)
        {
            data[i] = 0;
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }

    {
        GLuint data_buffer;
        glGenBuffers(1, &data_buffer);
        glBindBuffer( GL_SHADER_STORAGE_BUFFER, data_buffer );
        glBufferData( GL_SHADER_STORAGE_BUFFER, sizeof(float) * N, NULL, GL_STATIC_DRAW );
        glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, data_buffer );

        glGetError()  |  equals(GL_NO_ERROR, ERROR_MSG("An error!"));

        {
            float *data = (float*) glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, N * sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
            for (int i = 0; i < N; ++i)
            {
                data[i] = 1.0*rand()/RAND_MAX;
            }
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
    }

    {
        GLuint data_buffer;
        glGenBuffers(1, &data_buffer);
        glBindBuffer( GL_SHADER_STORAGE_BUFFER, data_buffer );
        glBufferData( GL_SHADER_STORAGE_BUFFER, sizeof(float) * N, NULL, GL_STATIC_DRAW );
        glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 6, data_buffer );

        glGetError()  |  equals(GL_NO_ERROR, ERROR_MSG("An error!"));

        {
            float *data = (float*) glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, N * sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
            for (int i = 0; i < N; ++i)
            {
                data[i] = 1.0*rand()/RAND_MAX;
            }
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, compute_shader);
    link_program(program);

    glUseProgram(program);

    glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, buffer );

    glGetError()  |  equals(GL_NO_ERROR, ERROR_MSG("An error!"));

    GLuint begin = glutGet(GLUT_ELAPSED_TIME);

    glDispatchCompute( N/1000, 1, 1 );
    glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );


    glGetError()  |  equals(GL_NO_ERROR, ERROR_MSG("An error!"));


    {
        float *data = (float*) glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, N * sizeof(float), GL_MAP_READ_BIT);

        GLuint end = glutGet(GLUT_ELAPSED_TIME);
        std::cout << "Time = " << (end-begin) << "ms" << std::endl;

        for (int i = 0; i < 10; ++i)
        {
            std::cout << i << " = " << data[i] << std::endl;
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }

    system("pause");

    return EXIT_SUCCESS;
}
catch(std::exception const &ex)
{
    std::cerr << "Failed with error: <" << typeid(ex).name() << ">: " << ex.what() << std::endl;
    return EXIT_FAILURE;
}