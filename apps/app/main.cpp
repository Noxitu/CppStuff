#include <iostream>
#include <noxitu/common/Factory.hpp>

int main() try
{
    std::string name = "impl2";
    std::cout << "Hello World" << std::endl;

    noxitu::common::Interface *op = noxitu::common::Factory<noxitu::common::Interface>().create(name);
    std::cout << "Inteface = " << op << std::endl;

    std::cout << (*op)() << std::endl;
}
catch(std::exception &ex)
{
    std::cerr << "Exception: " << ex.what() << std::endl;
    return EXIT_FAILURE;
}