/**********************************************************************

  Audacity: A Digital Audio Editor

  MeterToolBar.cpp

  Dominic Mazzoni
  Leland Lucius

  See MeterToolBar.h for details

*******************************************************************//*!

\class MeterToolBar
\brief A ToolBar that holds the VU Meter

*//*******************************************************************/



#include "MeterToolBar.h"
#include "widgets/AButton.h"

// For compilers that support precompilation, includes "wx/wx.h".
#include <wx/wxprec.h>

#include <wx/setup.h> // for wxUSE_* macros

#ifndef WX_PRECOMP
#include <wx/event.h>
#include <wx/intl.h>
#include <wx/tooltip.h>
#endif

#include <wx/gbsizer.h>

#include "AllThemeResources.h"
#include "Decibels.h"
#include "ToolManager.h"
#include "../ProjectAudioIO.h"
#include "../widgets/MeterPanel.h"

#if wxUSE_ACCESSIBILITY

class MeterButtonAx : public AButtonAx
{
   wxWeakRef<MeterPanel> mAssociatedMeterPanel{nullptr};
   bool mAccSilent{false};
public:

   MeterButtonAx(AButton& button, MeterPanel& panel)
      : AButtonAx(&button)
      , mAssociatedMeterPanel(&panel)
   { }

   wxAccStatus GetName(int childId, wxString* name) override
   {
      if(childId != wxACC_SELF)
         return wxACC_NOT_IMPLEMENTED;

      if(mAccSilent)
         *name = wxEmptyString; // Jaws reads nothing, and nvda reads "unknown"
      else
      {
         const auto button = static_cast<AButton*>(GetWindow());

         *name = button->GetName();

         if(const auto panel = mAssociatedMeterPanel.get())
         {
            // translations of strings such as " Monitoring " did not
            // always retain the leading space. Therefore a space has
            // been added to ensure at least one space, and stop
            // words from being merged
            if(panel->IsMonitoring())
               *name += wxT(" ") + _(" Monitoring ");
            else if(panel->IsActive())
               *name += wxT(" ") + _(" Active ");

            const auto dbRange = panel->GetDBRange();
            if (dbRange != -1)
               *name += wxT(" ") + wxString::Format(_(" Peak %2.f dB"), panel->GetPeakHold() * dbRange - dbRange);
            else
               *name += wxT(" ") + wxString::Format(_(" Peak %.2f "), panel->GetPeakHold());

            if(panel->IsClipping())
               *name += wxT(" ") + _(" Clipped ");
         }
      }
      return wxACC_OK;
   }

   wxAccStatus GetRole(int childId, wxAccRole* role) override
   {
      if(childId != wxACC_SELF)
         return wxACC_NOT_IMPLEMENTED;

      if(mAccSilent)
         *role = wxROLE_NONE; // Jaws and nvda both read nothing 
      else
         *role = wxROLE_SYSTEM_PUSHBUTTON;
      return wxACC_OK;
   }

   void SetSilent(bool silent)
   {
      if(mAccSilent != silent)
      {
         mAccSilent = silent;
         if(!silent)
         {
            NotifyEvent(
               wxACC_EVENT_OBJECT_FOCUS,
               GetWindow(),
               wxOBJID_CLIENT,
               wxACC_SELF);
         }
      }
   }
};

#endif

IMPLEMENT_CLASS(MeterToolBar, ToolBar);

////////////////////////////////////////////////////////////
/// Methods for MeterToolBar
////////////////////////////////////////////////////////////

BEGIN_EVENT_TABLE( MeterToolBar, ToolBar )
   EVT_SIZE( MeterToolBar::OnSize )
END_EVENT_TABLE()

//Standard constructor
MeterToolBar::MeterToolBar(AudacityProject &project, int type)
: ToolBar(project, type, XO("Combined Meter"), wxT("CombinedMeter"), true)
{
   if( mType == RecordMeterBarID ){
      mWhichMeters = kWithRecordMeter;
      mLabel = XO("Recording Meter");
      mSection = wxT("RecordMeter");
   } else if( mType == PlayMeterBarID ){
      mWhichMeters = kWithPlayMeter;
      mLabel = XO("Playback Meter");
      mSection = wxT("PlayMeter");
   } else {
      mWhichMeters = kWithPlayMeter | kWithRecordMeter;
   }
}

MeterToolBar::~MeterToolBar()
{
}

MeterToolBars MeterToolBar::GetToolBars(AudacityProject &project)
{
   return MeterToolBars{
      Get(project, true),
      Get(project, false)
   };
}

ConstMeterToolBars MeterToolBar::GetToolBars(const AudacityProject &project)
{
   return ConstMeterToolBars{
      Get(project, true),
      Get(project, false)
   };
}

MeterToolBar & MeterToolBar::Get(AudacityProject &project, bool forPlayMeterToolBar)
{
   auto& toolManager = ToolManager::Get(project);
   auto  toolBarID = forPlayMeterToolBar ? PlayMeterBarID : RecordMeterBarID;

   return *static_cast<MeterToolBar*>(toolManager.GetToolBar(toolBarID));
}

const MeterToolBar & MeterToolBar::Get(const AudacityProject &project, bool forPlayMeterToolBar)
{
   return Get( const_cast<AudacityProject&>( project ), forPlayMeterToolBar );
}

void MeterToolBar::Create(wxWindow * parent)
{
   ToolBar::Create(parent);

   UpdatePrefs();

   // Simulate a size event to set initial meter placement/size
   wxSizeEvent dummy;
   OnSize(dummy);
}

void MeterToolBar::ReCreateButtons()
{
   MeterPanel::State playState{ false }, recordState{ false };

   auto &projectAudioIO = ProjectAudioIO::Get( mProject );
   if (mPlayMeter &&
      projectAudioIO.GetPlaybackMeter() == mPlayMeter->GetMeter())
   {
      playState = mPlayMeter->SaveState();
      projectAudioIO.SetPlaybackMeter( nullptr );
   }

   if (mRecordMeter &&
      projectAudioIO.GetCaptureMeter() == mRecordMeter->GetMeter())
   {
      recordState = mRecordMeter->SaveState();
      projectAudioIO.SetCaptureMeter( nullptr );
   }

   ToolBar::ReCreateButtons();

   mPlayMeter->RestoreState(playState);
   if( playState.mSaved  ){
      projectAudioIO.SetPlaybackMeter( mPlayMeter->GetMeter() );
   }
   mRecordMeter->RestoreState(recordState);
   if( recordState.mSaved ){
      projectAudioIO.SetCaptureMeter( mRecordMeter->GetMeter() );
   }
}

void MeterToolBar::Populate()
{
   MakeButtonBackgroundsSmall();
   SetBackgroundColour( theTheme.Colour( clrMedium  ) );

   if( mWhichMeters & kWithRecordMeter ){
      //JKC: Record on left, playback on right.  Left to right flow
      //(maybe we should do it differently for Arabic language :-)  )
      mRecordSetupButton = safenew AButton(this);
      mRecordSetupButton->SetLabel({});
      mRecordSetupButton->SetName(_("Record Meter"));
      mRecordSetupButton->SetImages(
         theTheme.Image(bmpRecoloredUpSmall),
         theTheme.Image(bmpRecoloredUpHiliteSmall),
         theTheme.Image(bmpRecoloredDownSmall),
         theTheme.Image(bmpRecoloredHiliteSmall),
         theTheme.Image(bmpRecoloredUpSmall));
      mRecordSetupButton->SetIcon(theTheme.Image(bmpMic));
      mRecordSetupButton->SetButtonType(AButton::Type::FrameButton);
      mRecordSetupButton->SetMinSize({toolbarSingle, toolbarSingle});

      mRecordMeter = safenew MeterPanel( &mProject,
                                this,
                                wxID_ANY,
                                true,
                                wxDefaultPosition,
                                wxSize( 260, toolbarSingle) );
      /* i18n-hint: (noun) The meter that shows the loudness of the audio being recorded.*/
      mRecordMeter->SetName( XO("Recording Level"));
      /* i18n-hint: (noun) The meter that shows the loudness of the audio being recorded.
       This is the name used in screen reader software, where having 'Meter' first
       apparently is helpful to partially sighted people.  */
      mRecordMeter->SetLabel( XO("Meter-Record") );

#if wxUSE_ACCESSIBILITY
      auto meterButtonAcc = safenew MeterButtonAx(*mRecordSetupButton, *mRecordMeter);
#endif
      mRecordSetupButton->Bind(wxEVT_BUTTON, [=](wxCommandEvent&)
      {
#if wxUSE_ACCESSIBILITY
         meterButtonAcc->SetSilent(true);
#endif
         mRecordMeter->ShowMenu(
            mRecordMeter->ScreenToClient(
               ClientToScreen(mRecordSetupButton->GetClientRect().GetBottomLeft())
            )
         );
#if wxUSE_ACCESSIBILITY
         /* if stop/start monitoring was chosen in the menu, then by this point
         OnMonitoring has been called and variables which affect the accessibility
         name have been updated so it's now ok for screen readers to read the name of
         the button */
         meterButtonAcc->SetSilent(false);
#endif
      });
   }

   if( mWhichMeters & kWithPlayMeter ){
      mPlaySetupButton = safenew AButton(this);
      mPlaySetupButton->SetLabel({});
      mPlaySetupButton->SetName(_("Playback Meter"));
      mPlaySetupButton->SetImages(
         theTheme.Image(bmpRecoloredUpSmall),
         theTheme.Image(bmpRecoloredUpHiliteSmall),
         theTheme.Image(bmpRecoloredDownSmall),
         theTheme.Image(bmpRecoloredHiliteSmall),
         theTheme.Image(bmpRecoloredUpSmall));
      mPlaySetupButton->SetIcon(theTheme.Image(bmpSpeaker));
      mPlaySetupButton->SetButtonType(AButton::Type::FrameButton);
      mPlaySetupButton->SetMinSize({toolbarSingle, toolbarSingle});


      mPlayMeter = safenew MeterPanel( &mProject,
                              this,
                              wxID_ANY,
                              false,
                              wxDefaultPosition,
                              wxSize( 260, toolbarSingle ) );
      /* i18n-hint: (noun) The meter that shows the loudness of the audio playing.*/
      mPlayMeter->SetName( XO("Playback Level"));
      /* i18n-hint: (noun) The meter that shows the loudness of the audio playing.
       This is the name used in screen reader software, where having 'Meter' first
       apparently is helpful to partially sighted people.  */
      mPlayMeter->SetLabel( XO("Meter-Play"));

#if wxUSE_ACCESSIBILITY
      auto meterButtonAcc = safenew MeterButtonAx(*mPlaySetupButton, *mPlayMeter);
#endif
      mPlaySetupButton->Bind(wxEVT_BUTTON, [=](wxCommandEvent&)
      {
#if wxUSE_ACCESSIBILITY
         meterButtonAcc->SetSilent(true);
#endif
         mPlayMeter->ShowMenu(
            mPlayMeter->ScreenToClient(
               ClientToScreen(mPlaySetupButton->GetClientRect().GetBottomLeft())
            )
         );
#if wxUSE_ACCESSIBILITY
         meterButtonAcc->SetSilent(false);
#endif
      });
   }

   RebuildLayout(true);

   RegenerateTooltips();

   Layout();
}

void MeterToolBar::UpdatePrefs()
{
   RegenerateTooltips();

   // Set label to pull in language change
   SetLabel(XO("Meter"));

   // Give base class a chance
   ToolBar::UpdatePrefs();
}

void MeterToolBar::UpdateControls()
{
   if ( mPlayMeter )
      mPlayMeter->UpdateSliderControl();

   if ( mRecordMeter )
      mRecordMeter->UpdateSliderControl();
}

void MeterToolBar::RebuildLayout(bool force)
{
   const auto size = GetSize();
   const auto isHorizontal = size.x > size.y;

   if(!force)
   {
      const auto sizerOrientation = mWhichMeters == kCombinedMeter
         ? (isHorizontal ? wxVERTICAL : wxHORIZONTAL)
         : (isHorizontal ? wxHORIZONTAL : wxVERTICAL);
      
      if(mRootSizer->GetOrientation() == sizerOrientation)
      {
         Layout();
         return;
      }
   }

   if(mRootSizer != nullptr)
      GetSizer()->Remove(mRootSizer);

   std::unique_ptr<wxBoxSizer> playBarSizer;
   std::unique_ptr<wxBoxSizer> recordBarSizer;
   if(mWhichMeters & kWithPlayMeter)
   {
      playBarSizer = std::make_unique<wxBoxSizer>(isHorizontal ? wxHORIZONTAL : wxVERTICAL);
      playBarSizer->Add(mPlaySetupButton, 0, wxEXPAND);
      playBarSizer->Add(mPlayMeter, 1, wxEXPAND);
   }
   if(mWhichMeters & kWithRecordMeter)
   {
      recordBarSizer = std::make_unique<wxBoxSizer>(isHorizontal ? wxHORIZONTAL : wxVERTICAL);
      recordBarSizer->Add(mRecordSetupButton, 0, wxEXPAND);
      recordBarSizer->Add(mRecordMeter, 1, wxEXPAND);
   }

   if(playBarSizer && recordBarSizer)
   {
      Add(mRootSizer = safenew wxBoxSizer(isHorizontal ? wxVERTICAL : wxHORIZONTAL), 1, wxEXPAND);
      mRootSizer->Add(playBarSizer.release());
      mRootSizer->Add(recordBarSizer.release());
   }
   else if(playBarSizer)
      Add(mRootSizer = playBarSizer.release(), 1, wxEXPAND);
   else if(recordBarSizer)
      Add(mRootSizer = recordBarSizer.release(), 1, wxEXPAND);
}


void MeterToolBar::OnSize( wxSizeEvent & event)
{
   event.Skip();

   if(mRootSizer == nullptr)
      return;// We can be resized before populating...protect against it
   RebuildLayout(false);
}

bool MeterToolBar::Expose( bool show )
{
   auto &projectAudioIO = ProjectAudioIO::Get( mProject );
   if( show ) {
      if( mPlayMeter ) {
         projectAudioIO.SetPlaybackMeter( mPlayMeter->GetMeter() );
      }

      if( mRecordMeter ) {
         projectAudioIO.SetCaptureMeter( mRecordMeter->GetMeter() );
      }
   } else {
      if( mPlayMeter &&
         projectAudioIO.GetPlaybackMeter() == mPlayMeter->GetMeter() ) {
         projectAudioIO.SetPlaybackMeter( nullptr );
      }

      if( mRecordMeter &&
         projectAudioIO.GetCaptureMeter() == mRecordMeter->GetMeter() ) {
         projectAudioIO.SetCaptureMeter( nullptr );
      }
   }

   return ToolBar::Expose( show );
}

int MeterToolBar::GetInitialWidth()
{
   return (mWhichMeters ==
      (kWithRecordMeter + kWithPlayMeter)) ? 338 : 290;
}

// The meter's sizing code does not take account of the resizer
// Hence after docking we need to enlarge the bar (using fit)
// so that the resizer can be reached.
void MeterToolBar::SetDocked(ToolDock *dock, bool pushed) {
   ToolBar::SetDocked(dock, pushed);
   Fit();
}

void MeterToolBar::ShowOutputGainDialog()
{
   mPlayMeter->ShowDialog();
   mPlayMeter->UpdateSliderControl();
}

void MeterToolBar::ShowInputGainDialog()
{
   mRecordMeter->ShowDialog();
   mRecordMeter->UpdateSliderControl();
}

void MeterToolBar::AdjustOutputGain(int adj)
{
   if (adj < 0) {
      mPlayMeter->Decrease(-adj);
   }
   else {
      mPlayMeter->Increase(adj);
   }

   mPlayMeter->UpdateSliderControl();
}

void MeterToolBar::AdjustInputGain(int adj)
{
   if (adj < 0) {
      mRecordMeter->Decrease(-adj);
   }
   else {
      mRecordMeter->Increase(adj);
   }

   mRecordMeter->UpdateSliderControl();
}

static RegisteredToolbarFactory factory1{ RecordMeterBarID,
   []( AudacityProject &project ){
      return ToolBar::Holder{
         safenew MeterToolBar{ project, RecordMeterBarID } }; }
};
static RegisteredToolbarFactory factory2{ PlayMeterBarID,
   []( AudacityProject &project ){
      return ToolBar::Holder{
         safenew MeterToolBar{ project, PlayMeterBarID } }; }
};
static RegisteredToolbarFactory factory3{ MeterBarID,
   []( AudacityProject &project ){
      return ToolBar::Holder{
         safenew MeterToolBar{ project, MeterBarID } }; }
};

#include "ToolManager.h"

namespace {
AttachedToolBarMenuItem sAttachment1{
   /* i18n-hint: Clicking this menu item shows the toolbar
      with the recording level meters */
   RecordMeterBarID, wxT("ShowRecordMeterTB"), XXO("&Recording Meter Toolbar"),
   {}, { MeterBarID }
};
AttachedToolBarMenuItem sAttachment2{
   /* i18n-hint: Clicking this menu item shows the toolbar
      with the playback level meter */
   PlayMeterBarID, wxT("ShowPlayMeterTB"), XXO("&Playback Meter Toolbar"),
   {}, { MeterBarID }
};
//AttachedToolBarMenuItem sAttachment3{
//   /* --i18nhint: Clicking this menu item shows the toolbar
//      which has sound level meters */
//   MeterBarID, wxT("ShowMeterTB"), XXO("Co&mbined Meter Toolbar"),
//   { Registry::OrderingHint::After, "ShowPlayMeterTB" },
//   { PlayMeterBarID, RecordMeterBarID }
//};
}
