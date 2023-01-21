/**********************************************************************

  Audacity: A Digital Audio Editor

  @file EffectStage.h
  @brief decorator of a Sink with a non-time-warping effect

  Dominic Mazzoni
  Vaughan Johnson

  Paul Licameli split from Effect.h

**********************************************************************/

#ifndef __AUDACITY_EFFECT_STAGE__
#define __AUDACITY_EFFECT_STAGE__

#include "AudioGraphSource.h" // to inherit
#include "EffectInterface.h"
#include "SampleCount.h"
#include <functional>

class Track;

namespace AudioGraph {

//! Decorates a source with a non-timewarping effect, which may have latency
class AUDIO_GRAPH_API EffectStage final : public Source {
   // To force usage of Create() instead
   struct CreateToken {};
public:
   using Factory = std::function<std::shared_ptr<EffectInstanceEx>()>;

   //! Don't call directly but use Create()
   /*!
    @param factory used only in construction, will be invoked one or more times
    @pre `upstream.AcceptsBlockSize(inBuffers.BlockSize())`
    @post `AcceptsBlockSize(inBuffers.BlockSize())`
    @post `ProcessInitialize()` succeeded on each instance that was made by
       `factory`
    @param map not required after construction
    */
   EffectStage(CreateToken, bool multi, Source &upstream, Buffers &inBuffers,
      const Factory &factory, EffectSettings &settings,
      double sampleRate,
      std::optional<sampleCount> genLength, const Track &track);

   //! Satisfies postcondition of constructor or returns null
   static std::unique_ptr<EffectStage> Create(bool multi,
      Source &upstream, Buffers &inBuffers,
      const Factory &factory, EffectSettings &settings,
      double sampleRate,
      std::optional<sampleCount> genLength, const Track &track);

   EffectStage(const EffectStage&) = delete;
   EffectStage &operator =(const EffectStage &) = delete;
   //! Finalizes the instance
   ~EffectStage() override;

   /*!
    @return true
    */
   bool AcceptsBuffers(const Buffers &buffers) const override;
   //! See postcondition of constructor
   bool AcceptsBlockSize(size_t size) const override;

   std::optional<size_t> Acquire(Buffers &data, size_t bound) override;
   sampleCount Remaining() const override;
   bool Release() override;

private:
   sampleCount DelayRemaining() const
      { return std::max<sampleCount>(0, mDelayRemaining); }

   //! Produce exactly `curBlockSize` samples in `data`
   /*!
    @pre curBlockSize <= data.BlockSize() - outBufferOffset
    @pre curBlockSize <= data.Remaining() - outBufferOffset
    @pre curBlockSize <= mInBuffers.Remaining()
    @return success
    */
   bool Process(EffectInstanceEx &instance, size_t channel,
      const Buffers &data, size_t curBlockSize, size_t outBufferOffset) const;

   [[nodiscard]] std::optional<size_t> FetchProcessAndAdvance(
      Buffers &data, size_t bound, bool doZeros, size_t outBufferOffset = 0);

   Source &mUpstream;
   //! @invariant mInBuffers.BlockSize() <= mInBuffers.Remaining()
   Buffers &mInBuffers;
   const std::vector<std::shared_ptr<EffectInstanceEx>> mInstances;
   EffectSettings &mSettings;
   const double mSampleRate;
   const bool mIsProcessor;

   sampleCount mDelayRemaining;
   size_t mLastProduced{};
   size_t mLastZeroes{};
   bool mLatencyDone{ false };
   bool mCleared{ false };
};

/*
 @param multichannel true only when effect does not process each channel
    of track independently
 @param[out] map terminated with ChannelNameEOL
 */
AUDIO_GRAPH_API
unsigned MakeChannelMap(const Track &track, bool multichannel,
   // TODO: more-than-two-channels
   ChannelName map[3]);
}
#endif
