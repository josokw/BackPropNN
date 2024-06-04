#ifndef OSSTATE_H
#define OSSTATE_H

#include <ostream>

/// Responsible for storing the output stream flags.
/// The destructor restores the flags (RAII).
/// OSstate objects are not copyable.
class OSstate
{
public:
   explicit OSstate(std::ostream &os)
      : os_(os)
      , flags_(os.flags())
   {
   }
   ~OSstate() { os_.flags(flags_); }

   OSstate(const OSstate &rhs) = delete;
   OSstate &operator=(const OSstate &rhs) = delete;

private:
   std::ostream &os_;
   std::ios_base::fmtflags flags_;
};

#endif // OSSTATE_H
