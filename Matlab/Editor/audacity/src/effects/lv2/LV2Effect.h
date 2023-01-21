/**********************************************************************

  Audacity: A Digital Audio Editor

  LV2Effect.h

  Audacity(R) is copyright (c) 1999-2013 Audacity Team.
  License: GPL v2 or later.  See License.txt.

*********************************************************************/

#ifndef __AUDACITY_LV2_EFFECT__
#define __AUDACITY_LV2_EFFECT__

#if USE_LV2

class wxArrayString;

#include "LV2FeaturesList.h"
#include "LV2Ports.h"
#include "../../ShuttleGui.h"
#include "SampleFormat.h"
#include "../PerTrackEffect.h"

// We use deprecated LV2 interfaces to remain compatible with older
// plug-ins, so disable warnings
LV2_DISABLE_DEPRECATION_WARNINGS

#define LV2EFFECTS_VERSION wxT("1.0.0.0")
/* i18n-hint: abbreviates
   "Linux Audio Developer's Simple Plugin API (LADSPA) version 2" */
#define LV2EFFECTS_FAMILY XO("LV2")

class LV2Validator;

class LV2Effect final : public PerTrackEffect
{

   friend class LV2PluginValidator;

public:
   LV2Effect(const LilvPlugin &plug);
   virtual ~LV2Effect();

   // ComponentInterface implementation

   PluginPath GetPath() const override;
   ComponentInterfaceSymbol GetSymbol() const override;
   VendorSymbol GetVendor() const override;
   wxString GetVersion() const override;
   TranslatableString GetDescription() const override;

   // EffectDefinitionInterface implementation

   EffectType GetType() const override;
   EffectFamilySymbol GetFamily() const override;
   bool IsInteractive() const override;
   bool IsDefault() const override;
   RealtimeSince RealtimeSupport() const override;
   bool SupportsAutomation() const override;

   bool SaveSettings(
      const EffectSettings &settings, CommandParameters & parms) const override;
   bool LoadSettings(
      const CommandParameters & parms, EffectSettings &settings) const override;

   OptionalMessage LoadUserPreset(
      const RegistryPath & name, EffectSettings &settings) const override;
   bool SaveUserPreset(
      const RegistryPath & name, const EffectSettings &settings) const override;

   RegistryPaths GetFactoryPresets() const override;
   OptionalMessage LoadFactoryPreset(int id, EffectSettings &settings)
      const override;

   int ShowClientInterface(wxWindow &parent, wxDialog &dialog,
      EffectUIValidator *pValidator, bool forceModal) override;

   bool InitializePlugin();

   // EffectUIClientInterface implementation

   std::shared_ptr<EffectInstance> MakeInstance() const override;
   std::unique_ptr<EffectUIValidator> PopulateUI(
      ShuttleGui &S, EffectInstance &instance, EffectSettingsAccess &access,
      const EffectOutputs *pOutputs) override;
   bool CloseUI() override;

   bool CanExportPresets() override;
   void ExportPresets(const EffectSettings &settings) const override;
   OptionalMessage ImportPresets(EffectSettings &settings) override;

   bool HasOptions() override;
   void ShowOptions() override;

   // LV2Effect implementation

private:
   EffectSettings MakeSettings() const override;
   bool CopySettingsContents(
      const EffectSettings &src, EffectSettings &dst) const override;

   std::unique_ptr<EffectOutputs> MakeOutputs() const override;

   OptionalMessage LoadParameters(
      const RegistryPath & group, EffectSettings &settings) const;
   bool SaveParameters(
      const RegistryPath & group, const EffectSettings &settings) const;

private:
   const LilvPlugin &mPlug;
   const LV2FeaturesList mFeatures{ mPlug };

   const LV2Ports mPorts{ mPlug };

   bool mWantsOptionsInterface{ false };
   bool mWantsStateInterface{ false };

   size_t mFramePos{};

   FloatBuffers mCVInBuffers;
   FloatBuffers mCVOutBuffers;

   double mLength{};

   wxWindow *mParent{};

   // Mutable cache fields computed once on demand
   mutable bool mFactoryPresetsLoaded{ false };
   mutable RegistryPaths mFactoryPresetNames;
   mutable wxArrayString mFactoryPresetUris;
};

#endif
#endif
