/*!********************************************************************

  Audacity: A Digital Audio Editor

  @file AudioUnitWrapper.h

  Dominic Mazzoni
  Leland Lucius

  Paul Licameli split from AudioUnitEffect.h

**********************************************************************/
#ifndef AUDACITY_AUDIOUNIT_WRAPPER_H
#define AUDACITY_AUDIOUNIT_WRAPPER_H

#if USE_AUDIO_UNITS

#include <optional>
#include <map>
#include <set>
#include <unordered_map>
#include <wx/string.h>

#include "AudioUnitUtils.h"

class wxCFStringRef;
class wxMemoryBuffer;
class EffectSettings;
class TranslatableString;
class AudioUnitWrapper;

//! This works as a cached copy of state stored in an AudioUnit, but can also
//! outlive it
struct AudioUnitEffectSettings {
   //! Object from which settings were fetched
   const AudioUnitWrapper *pSource{};

   //! The effect object and all Settings objects coming from it share this
   //! set of strings, which allows Pair below to copy without allocations.
   /*!
    Note that names associated with parameter IDs are not invariant metadata
    of an AudioUnit effect!  The names can themselves depend on the current
    values.  Example:  AUGraphicEQ changes names of slider parameters when you
    change the switch between 10 and 31 bands.
    */
   using StringSet = std::set<wxString>;
   const std::shared_ptr<StringSet> mSharedNames{
      std::make_shared<StringSet>() };
   
   //! Map from numerical parameter IDs (not always a small initial segment
   //! of the integers) to optional pairs of names and floating point values
   using Pair = std::pair<const wxString &, AudioUnitParameterValue>;
   using Map = std::map<AudioUnitParameterID, std::optional<Pair>>;
   Map values;

   AudioUnitEffectSettings() = default;
   AudioUnitEffectSettings(Map map) : values{ move(map) } {}
   
   //! Get a pointer to a durable copy of `name`
   //! May allocate memory
   const wxString &Intern(const wxString &name) {
      // std::set::insert guarantees this iterator is not at the end
      auto [iter, _] = mSharedNames->insert(name);
      // so dereference it merrily
      return *iter;
   }

   //! Associate nullopt with all keys already present in the map
   void ResetValues()
   {
      for (auto &[_, value] : values)
         value.reset();
   }
};

//! Common base class for AudioUnitEffect and its Instance
/*!
 Maintains a smart handle to an AudioUnit (also called AudioComponentInstance)
 in the SDK and defines some utility functions
 */
struct AudioUnitWrapper
{
   using Parameters = PackedArray::Ptr<const AudioUnitParameterID>;

   static AudioUnitEffectSettings &GetSettings(EffectSettings &settings);
   static const AudioUnitEffectSettings &GetSettings(
      const EffectSettings &settings);

   /*!
    @param pParameters if non-null, use those; else, fetch from the AudioUnit
    */
   AudioUnitWrapper(AudioComponent component, Parameters *pParameters)
      : mComponent{ component }
      , mParameters{ pParameters ? *pParameters : mOwnParameters }
   {
   }

   // Supply most often used values as defaults for scope and element
   template<typename T>
   OSStatus GetFixedSizeProperty(AudioUnitPropertyID inID, T &property,
      AudioUnitScope inScope = kAudioUnitScope_Global,
      AudioUnitElement inElement = 0) const
   {
      // Supply mUnit.get() to the non-member function
      return AudioUnitUtils::GetFixedSizeProperty(mUnit.get(),
         inID, property, inScope, inElement);
   }

   // Supply most often used values as defaults for scope and element
   template<typename T>
   OSStatus GetVariableSizeProperty(AudioUnitPropertyID inID,
      PackedArray::Ptr<T> &pObject,
      AudioUnitScope inScope = kAudioUnitScope_Global,
      AudioUnitElement inElement = 0) const
   {
      return AudioUnitUtils::GetVariableSizeProperty(mUnit.get(),
         inID, pObject, inScope, inElement);
   }

   // Supply most often used values as defaults for scope and element
   template<typename T>
   OSStatus SetProperty(AudioUnitPropertyID inID, const T &property,
      AudioUnitScope inScope = kAudioUnitScope_Global,
      AudioUnitElement inElement = 0) const
   {
      // Supply mUnit.get() to the non-member function
      return AudioUnitUtils::SetProperty(mUnit.get(),
         inID, property, inScope, inElement);
   }

   class ParameterInfo;
   //! Return value: if true, continue visiting
   using ParameterVisitor =
      std::function< bool(const ParameterInfo &pi, AudioUnitParameterID ID) >;
   void ForEachParameter(ParameterVisitor visitor) const;

   //! Obtain dump of the setting state of an AudioUnit instance
   /*!
    @param binary if false, then produce XML serialization instead; but
    AudioUnits does not need to be told the format again to reinterpret the blob
    @return smart pointer to data, and an error message
    */
   std::pair<CF_ptr<CFDataRef>, TranslatableString>
   MakeBlob(const AudioUnitEffectSettings &settings,
      const wxCFStringRef &cfname, bool binary) const;

   //! Interpret the dump made before by MakeBlob
   /*!
    @param group only for formatting error messages
    @return an error message
    */
   TranslatableString InterpretBlob(AudioUnitEffectSettings &settings,
      const wxString &group, const wxMemoryBuffer &buf) const;

   //! May allocate memory, so should be called only in the main thread
   bool FetchSettings(AudioUnitEffectSettings &settings) const;
   bool StoreSettings(const AudioUnitEffectSettings &settings) const;

   bool CreateAudioUnit();

   AudioUnit GetAudioUnit() const { return mUnit.get(); }
   AudioComponent GetComponent() const { return mComponent; }
   const Parameters &GetParameters() const
   { return mParameters; }

   // @param identifier only for logging messages
   bool SetRateAndChannels(double sampleRate, const wxString &identifier);

protected:
   const AudioComponent mComponent;
   AudioUnitCleanup<AudioUnit, AudioComponentInstanceDispose> mUnit;

   Parameters mOwnParameters;
   Parameters &mParameters;

   // Reassinged in GetRateAndChannels()
   unsigned mAudioIns{ 2 };
   unsigned mAudioOuts{ 2 };
};

class AudioUnitWrapper::ParameterInfo final
{
public:
   //! Make a structure holding a key for the config file and a value
   ParameterInfo(AudioUnit mUnit, AudioUnitParameterID parmID);
   //! Recover the parameter ID from the key, if well formed
   static std::optional<AudioUnitParameterID> ParseKey(const wxString &key);

   std::optional<wxString> mName;
   AudioUnitUtils::ParameterInfo mInfo{};

private:
   // constants
   static constexpr char idBeg = wxT('<');
   static constexpr char idSep = wxT(',');
   static constexpr char idEnd = wxT('>');
};

#endif

#endif
