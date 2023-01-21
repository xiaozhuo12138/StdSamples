/*  SPDX-License-Identifier: GPL-2.0-or-later */
/*!********************************************************************

  Audacity: A Digital Audio Editor

  LinkFailedDialog.cpp

  Dmitry Vedenko

**********************************************************************/
#include "LinkFailedDialog.h"

#include <wx/button.h>

#include "CodeConversions.h"
#include "ServiceConfig.h"

#include "ShuttleGui.h"

#include "widgets/HelpSystem.h"


namespace cloud::audiocom
{

LinkFailedDialog::LinkFailedDialog(wxWindow* parent)
    : wxDialogWrapper(
         parent, wxID_ANY, XO("Link account"), wxDefaultPosition, { 442, -1 },
         wxDEFAULT_DIALOG_STYLE)
{
   SetMinSize({ 442, -1 });
   ShuttleGui s(this, eIsCreating);

   s.StartVerticalLay();
   {
      s.StartInvisiblePanel(16);
      {
         s.SetBorder(0);

         s.AddFixedText(
            XO("We were unable to link your account. Please try again."),
            false, 410);

         s.AddSpace(0, 16, 0);

         s.StartHorizontalLay(wxEXPAND, 0);
         {
            s.AddSpace(1, 0, 1);

            s.AddButton(XO("&Cancel"))
               ->Bind(wxEVT_BUTTON, [this](auto) { EndModal(wxID_CANCEL); });
            
            auto btn = s.AddButton(XO("&Try again"));

            btn->Bind(
               wxEVT_BUTTON,
               [this](auto)
               {
                  OpenInDefaultBrowser({ audacity::ToWXString(
                     GetServiceConfig().GetOAuthLoginPage()) });
                  EndModal(wxID_RETRY);
               });

            btn->SetDefault();
         }
         s.EndHorizontalLay();
         
      }
      s.EndInvisiblePanel();
   }
   s.EndVerticalLay();

   Layout();
   Fit();
   Center();

   Bind(
      wxEVT_CHAR_HOOK,
      [this](auto& evt)
      {
         if (!IsEscapeKey(evt))
         {
            evt.Skip();
            return;
         }

         EndModal(wxID_CANCEL);
      });
}

LinkFailedDialog::~LinkFailedDialog()
{
}

} // namespace cloud::audiocom
