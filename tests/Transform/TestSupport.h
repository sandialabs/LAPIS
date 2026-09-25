#ifndef LAPIS_TESTS_TRANSFORM_TESTSUPPORT_H
#define LAPIS_TESTS_TRANSFORM_TESTSUPPORT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::lapis::test {

class TestContext {
public:
  template <typename Test> void run(llvm::StringRef name, Test test) {
    currentTest = name;
    const unsigned failuresBefore = failures;
    if (!test(*this) && failures == failuresBefore)
      fail("test returned failure without a diagnostic", __FILE__, __LINE__);
  }

  bool check(bool condition, llvm::StringRef expression, llvm::StringRef file,
             unsigned line) {
    if (condition)
      return true;
    fail((llvm::Twine("check failed: ") + expression).str(), file, line);
    return false;
  }

  void fail(llvm::StringRef message, llvm::StringRef file, unsigned line) {
    llvm::errs() << file << ':' << line << ": " << currentTest << ": "
                 << message << '\n';
    ++failures;
  }

  int getExitCode() const { return failures == 0 ? 0 : 1; }

private:
  llvm::StringRef currentTest = "unnamed test";
  unsigned failures = 0;
};

template <typename T>
bool requireExpected(TestContext &context, llvm::Expected<T> &value,
                     llvm::StringRef expression, llvm::StringRef file,
                     unsigned line) {
  if (value)
    return true;
  context.fail((llvm::Twine("unexpected error from ") + expression + ": " +
                llvm::toString(value.takeError()))
                   .str(),
               file, line);
  return false;
}

} // namespace mlir::lapis::test

#define LAPIS_CHECK(context, condition)                                        \
  (context).check(static_cast<bool>(condition), #condition, __FILE__, __LINE__)

#define LAPIS_REQUIRE(context, condition)                                      \
  do {                                                                         \
    if (!LAPIS_CHECK(context, condition))                                      \
      return false;                                                            \
  } while (false)

#define LAPIS_REQUIRE_EXPECTED(context, value)                                 \
  do {                                                                         \
    if (!::mlir::lapis::test::requireExpected(context, value, #value,          \
                                              __FILE__, __LINE__))             \
      return false;                                                            \
  } while (false)

#endif // LAPIS_TESTS_TRANSFORM_TESTSUPPORT_H
