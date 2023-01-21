/**********************************************************************

  Audacity: A Digital Audio Editor

  @file VST3Effect.h

  @author Vitaly Sverchinsky

  @brief Part of Audacity VST3 module

**********************************************************************/


#pragma once

#include <wx/wx.h>

#include <public.sdk/source/vst/hosting/module.h>

#include "effects/PerTrackEffect.h"

namespace Steinberg
{
   namespace Vst
   {
      class IEditController;
   }
}

class NumericTextCtrl;
class VST3Instance;

class VST3ParametersWindow;

/**
 * \brief Objects of this class connect Audacity with VST3 effects
 */
class VST3Effect final : public PerTrackEffect
{
   friend class VST3PluginValidator;

   // Keep strong reference to a module; this because it has to be destroyed in the destructor of this class,
   // otherwise the destruction of mEditController and mEffectComponent would trigger a memory fault.
   std::shared_ptr<VST3::Hosting::Module> mModule;
   const VST3::Hosting::ClassInfo mEffectClassInfo;

   wxWindow* mParent { nullptr };

   // Mutable cache fields computed once on demand
   mutable bool mRescanFactoryPresets { true };
   mutable RegistryPaths mFactoryPresets;

public:

   static EffectFamilySymbol GetFamilySymbol();

   VST3Effect(
      std::shared_ptr<VST3::Hosting::Module> module,
      VST3::Hosting::ClassInfo effectClassInfo);

   VST3Effect(const VST3Effect&) = delete;
   VST3Effect(VST3Effect&&) = delete;
   VST3Effect& operator=(const VST3Effect&) = delete;
   VST3Effect& operator=(VST3Effect&) = delete;
   
   ~VST3Effect() override;

   PluginPath GetPath() const override;
   ComponentInterfaceSymbol GetSymbol() const override;
   VendorSymbol GetVendor() const override;
   wxString GetVersion() const override;
   TranslatableString GetDescription() const override;

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

   EffectSettings MakeSettings() const override;
   bool CopySettingsContents(const EffectSettings& src, EffectSettings& dst) const override;

private:
   
   bool LoadPreset(const wxString& path, EffectSettings& settings) const;
};
