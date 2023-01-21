/*
 * Copyright (C) 2002 2003 2004, Magnus Hjorth
 *
 * This file is part of gWaveEdit.
 *
 * gWaveEdit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by        
 * the Free Software Foundation; either version 2 of the License, or  
 * (at your option) any later version.
 *
 * gWaveEdit is distributed in the hope that it will be useful,   
 * but WITHOUT ANY WARRANTY; without even the implied warranty of  
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gWaveEdit; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */


#ifndef VOLUME_DIALOG_H_INCLUDED
#define VOLUME_DIALOG_H_INCLUDED

#include <gtk/gtk.h>
#include "mainwindow.h"
#include "effectdialog.h"
#include "float_box.h"

#define VOLUME_DIALOG_TYPE          (volume_dialog_get_type ())
#define VOLUME_DIALOG(obj)          (G_TYPE_CHECK_INSTANCE_CAST ((obj), VOLUME_DIALOG_TYPE, VolumeDialog))
#define IS_VOLUME_DIALOG(obj)       (G_TYPE_CHECK_INSTANCE_TYPE ((obj), VOLUME_DIALOG_TYPE))
#define VOLUME_DIALOG_CLASS(klass)  (G_TYPE_CHECK_CLASS_CAST ((klass),  VOLUME_DIALOG_TYPE, VolumeDialogClass))

typedef struct _VolumeDialog {
     EffectDialog ed;
     Floatbox *start_percent, *end_percent;
} VolumeDialog;

typedef struct _VolumeDialogClass {
     EffectDialogClass ed_class;
} VolumeDialogClass;

GType volume_dialog_get_type(void);

#endif
