/***************************************************************************
 *   2007-2016 by Peter Semiletov                            *
 *   tea@list.ru                                             *
 
 this code is Public Domain
 
 ***************************************************************************/


#include "shortcuts.h"
#include "utils.h"
#include "gui_utils.h"

#include <QString>
#include <QDebug>


#include <QKeyEvent>
#include <QAction>
#include <QStringList>
#include <QKeySequence>




QString mod_to_string (Qt::KeyboardModifiers k)
{
  QString s;

  if (k & Qt::ShiftModifier)
     s += "Shift+";

  if (k & Qt::ControlModifier)
     s += "Ctrl+";

  if (k & Qt::AltModifier)
     s += "Alt+";

  if (k & Qt::MetaModifier)
     s+= "Meta+";

  return s;
}


QString keycode_to_string (int k)
{
//  return QKeySequence(k).toString();

  QString s;

  switch (k)
         {
          case Qt::Key_F1:
                          s = "F1";
                          break;

          case Qt::Key_F2:
                          s = "F2";
                          break;

          case Qt::Key_F3:
                          s = "F3";
                          break;

          case Qt::Key_F4:
                          s = "F4";
                          break;

          case Qt::Key_F5:
                          s = "F5";
                          break;

          case Qt::Key_F6:
                          s = "F6";
                          break;

          case Qt::Key_F7:
                          s = "F7";
                          break;

          case Qt::Key_F8:
                          s = "F8";
                          break;

          case Qt::Key_F9:
                          s = "F9";
                          break;

          case Qt::Key_F10:
                          s = "F10";
                          break;

          case Qt::Key_F11:
                          s = "F11";
                          break;

          case Qt::Key_F12:
                          s = "F12";
                          break;

          default:
                  s = QChar (k);
         }

  return s;
}



CShortcuts::CShortcuts (QWidget *widget)
{
  w = widget;
}


//FIXME:
void CShortcuts::captions_iterate()
{
  captions.clear();
  QList <QAction *> a = w->findChildren <QAction *>();
    
  for (auto *ac: a)
          if (ac)
          if (! ac->text().isEmpty())
             {
              captions.prepend (ac->text());
             // qDebug() << ac->text();
             }

 captions.sort();
 captions.removeDuplicates(); //nasty hack
}


QAction* CShortcuts::find_by_caption (const QString &text)
{
  QList<QAction *> a = w->findChildren<QAction *>();
  
  for (auto *ac: a)
          if (ac->text() == text)
             return ac;
 
  return NULL;
}


QAction* CShortcuts::find_by_shortcut (const QString &shcut)
{
  QList<QAction *> a = w->findChildren<QAction *>();

  for (auto *ac: a)
         if (ac->shortcut().toString() == shcut)
            return ac;
  
  return NULL;
}


QKeySequence CShortcuts::find_seq_by_caption (const QString &text)
{
  QAction *a = find_by_caption (text);

  if (a)
     return a->shortcut();

  return QKeySequence::fromString ("Ctrl+Alt+Z");
}


void CShortcuts::set_new_shortcut (const QString &menuitem, const QString &shcut)
{
  QAction *b = find_by_shortcut (shcut);
  if (b)
     b->setShortcut (QKeySequence(""));

  QAction *a = find_by_caption (menuitem);
  if (a)
     a->setShortcut (QKeySequence (shcut));
}


void CShortcuts::save_to_file (const QString &file_name)
{
  QList<QAction *> a = w->findChildren<QAction *>();
  QString s;

  for (auto *ac: a)
       if (! ac->shortcut().toString().isEmpty())
           s.append (ac->text()).append ("=").append (ac->shortcut().toString()).append ("\n");
                 
  qstring_save (file_name, s);
}


void CShortcuts::load_from_file (const QString &file_name)
{
  if (! file_exists (file_name))
     return;

  QHash <QString, QString> hash = hash_load_keyval (file_name);

  QList<QAction *> a = w->findChildren<QAction *>();

  for (auto *ac: a)
       if (hash.contains (ac->text()))
          ac->setShortcut (QKeySequence (hash.value (ac->text())));
}


void CShortcutEntry::keyPressEvent (QKeyEvent *event)
{
  event->accept();
  QString text = mod_to_string (event->modifiers()) + keycode_to_string (event->key());
  setText (text);
  qDebug() << text;
}
