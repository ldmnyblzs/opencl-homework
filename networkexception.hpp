#ifndef NETWORK_EXCEPTION_HPP
#define NETWORK_EXCEPTION_HPP

#include <stdexcept>

/*!
 * 
 */
class NetworkException : public std::runtime_error
{
public:
  NetworkException(const char* message) : std::runtime_error(message) {}
};

#endif // NETWORK_EXCEPTION_HPP
