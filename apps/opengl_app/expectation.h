#pragma once
#include <sstream>

class ExpectationSource
{
private:
    char const * const m_msg;
    char const * const m_file;
    int const m_line;
public:
    ExpectationSource(char const *msg, char const *file, int line) :
        m_msg(msg),
        m_file(file),
        m_line(line)
    {}

    std::string operator()() const
    {
        std::stringstream ss;
        ss << "Occured in " << m_msg << " in " << m_file << ":" << m_line;
        return ss.str();
    }
};

class Expectation
{
public:
    template<typename ValueType, typename ExpectationType>
    friend ValueType&& operator| (ValueType &&value, ExpectationType const &expectation)
    {
        expectation.validate(value);
        return std::forward<ValueType>(value);
    }
};

template<typename ExpectedValueType, typename Lambda>
class EqualsExpectation : public Expectation
{
private:
    ExpectedValueType const &m_expected_value;
    Lambda const &m_lambda;
public:
    EqualsExpectation(ExpectedValueType const &expected_value, Lambda const &lambda) :
        m_expected_value(expected_value),
        m_lambda(lambda)
    {}

    template<typename ValueType>
    void validate(ValueType &&value) const
    {
        if (value != m_expected_value)
        {
            std::stringstream ss;
            ss << "Unmet expectation: Got " << value << " instead of " << m_expected_value << ".";
            ss << " " << m_lambda();
            throw std::runtime_error(ss.str());
        }
    }
};

template<typename ExpectedValueType, typename Lambda>
auto equals(ExpectedValueType const &expected_value, Lambda lambda) -> EqualsExpectation<ExpectedValueType, Lambda>
{
    return {expected_value, lambda};
}




#define ERROR_MSG(msg) ExpectationSource(msg, __FILE__, __LINE__)