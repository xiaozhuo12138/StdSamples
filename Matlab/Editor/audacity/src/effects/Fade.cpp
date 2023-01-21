/**********************************************************************

  Audacity: A Digital Audio Editor

  Fade.cpp

  Robert Leidle

*******************************************************************//**

\class EffectFade
\brief An Effect that reduces the volume to zero over  achosen interval.

*//*******************************************************************/


#include "Fade.h"

#include <wx/intl.h>

#include "LoadEffects.h"

const ComponentInterfaceSymbol EffectFadeIn::Symbol
{ XO("Fade In") };

namespace{ BuiltinEffectsModule::Registration< EffectFadeIn > reg; }

const ComponentInterfaceSymbol EffectFadeOut::Symbol
{ XO("Fade Out") };

namespace{ BuiltinEffectsModule::Registration< EffectFadeOut > reg2; }

EffectFade::EffectFade(bool fadeIn)
{
   mFadeIn = fadeIn;
}

EffectFade::~EffectFade()
{
}

// ComponentInterface implementation

ComponentInterfaceSymbol EffectFade::GetSymbol() const
{
   return mFadeIn
      ? EffectFadeIn::Symbol
      : EffectFadeOut::Symbol;
}

TranslatableString EffectFade::GetDescription() const
{
   return mFadeIn
      ? XO("Applies a linear fade-in to the selected audio")
      : XO("Applies a linear fade-out to the selected audio");
}

// EffectDefinitionInterface implementation

EffectType EffectFade::GetType() const
{
   return EffectTypeProcess;
}

bool EffectFade::IsInteractive() const
{
   return false;
}

unsigned EffectFade::GetAudioInCount() const
{
   return 1;
}

unsigned EffectFade::GetAudioOutCount() const
{
   return 1;
}

bool EffectFade::ProcessInitialize(
   EffectSettings &, double, ChannelNames chanMap)
{
   mSample = 0;
   return true;
}

size_t EffectFade::ProcessBlock(EffectSettings &,
   const float *const *inBlock, float *const *outBlock, size_t blockLen)
{
   const float *ibuf = inBlock[0];
   float *obuf = outBlock[0];

   if (mFadeIn)
   {
      for (decltype(blockLen) i = 0; i < blockLen; i++)
      {
         obuf[i] =
            (ibuf[i] * ( mSample++ ).as_float()) /
            mSampleCnt.as_float();
      }
   }
   else
   {
      for (decltype(blockLen) i = 0; i < blockLen; i++)
      {
         obuf[i] = (ibuf[i] *
                    ( mSampleCnt - 1 - mSample++ ).as_float()) /
            mSampleCnt.as_float();
      }
   }

   return blockLen;
}
