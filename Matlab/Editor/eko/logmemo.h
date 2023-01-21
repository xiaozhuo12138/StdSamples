/***************************************************************************
 *   Copyright (C) 2007-2010 by Peter Semiletov                            *
 *   peter.semiletov@gmail.com                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef LOGMEMO_H
#define LOGMEMO_H

//#include <QtGui>

#include <QObject>
#include <QPlainTextEdit>
#include <QString>


class CLogMemo: public QObject
{
  Q_OBJECT

public:

  QPlainTextEdit *memo;
  CLogMemo (QPlainTextEdit *m);

public slots:

  void log (const QString &text);
};

#endif // LOGMEMO_H
