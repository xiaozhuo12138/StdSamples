/**********************************************************************

  Audacity: A Digital Audio Editor

  AVUtilImpl.cpp

  Dmitry Vedenko

**********************************************************************/

extern "C"
{
#include "../../avutil/56/avconfig.h"
#include "../../ffmpeg-4.2.4-single-header.h"
}

#include <wx/log.h>

#include "FFmpegFunctions.h"

#include "wrappers/AVFrameWrapper.h"

#include "../../FFmpegAPIResolver.h"
#include "../../FFmpegLog.h"

namespace avutil_56
{
#include "../AVFrameWrapperImpl.inl"
#include "../FFmpegLogImpl.inl"

const bool registered = ([]() {
   FFmpegAPIResolver::Get().AddAVUtilFactories(56, {
      &CreateAVFrameWrapper,
      &CreateLogCallbackSetter
   });

   return true;
})();
}
