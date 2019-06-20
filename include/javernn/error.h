#include <exception>
#include <string>

namespace javernn{

/**
 * error exception class for tiny-dnn
 **/
class Error : public std::exception {
 public:
  explicit Error(const std::string &msg) : msg_(msg) {}
  const char *what() const throw() override { return msg_.c_str(); }

 private:
  std::string msg_;
};

class NotimplementedError : public Error {
 public:
  explicit NotimplementedError(const std::string &msg = "not implemented")
    : Error(msg) {}
};

}  // namespace javernn