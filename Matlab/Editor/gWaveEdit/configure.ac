#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_INIT([gWaveEdit],[1.4.25pre],[])
AC_CONFIG_AUX_DIR([autoconf])
AC_CONFIG_MACRO_DIR([autoconf-m4])
AM_INIT_AUTOMAKE([1.10 -Wall foreign])

AC_CONFIG_HEADERS([config.h])
AM_GNU_GETTEXT([external])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
m4_ifdef([AM_MAINTAINER_MODE], [AM_MAINTAINER_MODE])

#L#T_INIT

# Set gettext package name
GETTEXT_PACKAGE=gwaveedit
AC_SUBST(GETTEXT_PACKAGE)
AC_DEFINE_UNQUOTED(GETTEXT_PACKAGE,"$GETTEXT_PACKAGE", [GETTEXT package name])

# Various options

AC_ARG_WITH(double-samples,
[  --with-double-samples   Use double precision floating point numbers for sound
                          processing (slows down program but may increase 
                          quality for 24/32-bit samplerates).],
AC_DEFINE([USE_DOUBLE_SAMPLES],[],
  [Use gdouble instead of gfloat as sample data type]) 
MHW_DOUBLE="yes")

AC_DEFUN([AC_ENABLE_WERROR],[
	if test "x$enableval" = "xyes" ; then
	AM_CFLAGS="$AM_CFLAGS -Werror"
	fi
])

AC_ARG_ENABLE(werror,
[  --enable-werror         Treat warnings as errors during compilation],
AC_ENABLE_WERROR)

DEFAULT_DRIVER=auto
AC_ARG_WITH(default-driver,
[  --with-default-driver=driver 
                           Choose default sound driver (one of oss,alsa,pulse,
                           jack,sun,pa,sdl,auto; default: auto)],
  DEFAULT_DRIVER=$withval)
AC_DEFINE_UNQUOTED(DEFAULT_DRIVER,"${DEFAULT_DRIVER}",[Default sound driver])

DEFAULT_LADSPA_PATH="/usr/lib/ladspa:/usr/local/lib/ladspa"
AC_ARG_WITH(default-ladspa-path,
[  --with-default-ladspa-path=path 
                           Path to use if LADSPA_PATH variable is not set 
                           (default: "/usr/lib/ladspa:/usr/local/lib/ladspa")],
DEFAULT_LADSPA_PATH=$withval)
AC_DEFINE_UNQUOTED(DEFAULT_LADSPA_PATH,"${DEFAULT_LADSPA_PATH}",[Default path for LADSPA plugins])

DEFAULT_MIXERAPP="xterm -e alsamixer"
AC_ARG_WITH(default-mixerapp,
[  --with-default-mixerapp=appname
                           Default command for "launch mixer" button
                           (default: xterm -e alsamixer)],
DEFAULT_MIXERAPP=$withval)
AC_DEFINE_UNQUOTED(DEFAULT_MIXERAPP,"${DEFAULT_MIXERAPP}",[Default mixer application])

# Check for GTK

AC_ARG_ENABLE(
	gtk3,
	AS_HELP_STRING([--enable-gtk3],[enable to use gtk-3.0 instead of gtk-2.0]),
	[case "${enableval}" in
		yes)  enable_gtk3=yes ;;
		no)   enable_gtk3=no ;;
		*) AC_MSG_ERROR([bad value "${enableval}" for --enable-gtk3, use "yes" (default) or "no".]) ;;
	esac],
	[]
)

if test "x$enable_gtk3" = "xyes" ; then
	gtk_modules="gtk+-3.0 >= 3.0.0"
else
	gtk_modules="gtk+-2.0 >= 2.14.0"
fi

PKG_CHECK_MODULES(GTK, [$gtk_modules])

AC_SUBST(GTK_CFLAGS)
AC_SUBST(GTK_LIBS)

# Disable gtk deprecations?

AC_ARG_ENABLE(
	gtk_warnings,
	[AS_HELP_STRING([--enable-gtk-warnings],[So you can see what is deprecated in GTK2...])],
	[case "${enableval}" in
		yes)  enable_gtk_warnings=yes ;;
		no)   enable_gtk_warnings=no ;;
	esac],
	[enable_gtk_warnings=no]
)

# Checks for programs.
AC_PROG_CC

dnl Checks for libraries.
dnl Replace main with a function in -libs:
AC_CHECK_LIB(ibs, main)
AC_CHECK_LIB(m, round)

dnl This is needed for PortAudio under GNU/Linux...
AC_CHECK_LIB(pthread,pthread_create)

dnl Check for libsndfile

HAVE_SNDFILE=yes
SKIP_SNDFILE_CHECK=no
SNDFILE_LIBS=""
SNDFILE_CFLAGS=""

AC_DEFUN([AC_WITH_LIBSNDFILE],[
	if test "x$withval" = "xno" ; then 
		HAVE_SNDFILE=no
	else
		SNDFILE_LIBS="-L$withval/lib -lsndfile -lm"
		SNDFILE_CFLAGS="-I$withval/include"
		SKIP_SNDFILE_CHECK=yes
	fi
])

AC_ARG_WITH(libsndfile,[  --with-libsndfile=prefix Prefix where libsndfile is installed (optional)
  --without-libsndfile    Do not use libsndfile library],AC_WITH_LIBSNDFILE)

if test "x$HAVE_SNDFILE" = "xyes" -a "x$SKIP_SNDFILE_CHECK" = "xno"; then
	PKG_CHECK_MODULES(SNDFILE, sndfile, , HAVE_SNDFILE=no)
fi

if test "x$HAVE_SNDFILE" = "xyes" ; then
	AC_DEFINE(HAVE_LIBSNDFILE,1,[Use libsndfile library])
fi

AC_SUBST(SNDFILE_LIBS)
AC_SUBST(SNDFILE_CFLAGS)

dnl Check for libsamplerate

HAVE_LIBSAMPLERATE=yes
SKIP_LIBSAMPLERATE_CHECK=no
LIBSAMPLERATE_LIBS=""
LIBSAMPLERATE_CFLAGS=""

AC_DEFUN([AC_WITH_LIBSAMPLERATE],[
	if test "x$withval" = "xno" ; then 
		HAVE_LIBSAMPLERATE=no
	else
		LIBSAMPLERATE_LIBS="-L$withval/lib -lsamplerate"
		LIBSAMPLERATE_CFLAGS="-I$withval/include"
		SKIP_LIBSAMPLERATE_CHECK=yes
	fi
])

AC_ARG_WITH(libsamplerate,[  --with-libsamplerate=prefix Prefix where libsamplerate is installed (optional)
  --without-libsamplerate Do not use libsamplerate library],AC_WITH_LIBSAMPLERATE)

if test "x$HAVE_LIBSAMPLERATE" = "xyes" -a "x$SKIP_LIBSAMPLERATE_CHECK" = "xno"; then
	PKG_CHECK_MODULES(LIBSAMPLERATE, samplerate, , HAVE_LIBSAMPLERATE=no)
fi

if test "x$HAVE_LIBSAMPLERATE" = "xyes" ; then
	AC_DEFINE(HAVE_LIBSAMPLERATE,1,[Use libsamplerate library])
fi

AC_SUBST(LIBSAMPLERATE_LIBS)
AC_SUBST(LIBSAMPLERATE_CFLAGS)

dnl Check for PortAudio

PORTAUDIO_LIBS=""
MHW_HAVE_PORTAUDIO=yes

AC_ARG_WITH(portaudio,[  --without-portaudio     Do not use portaudio library],
if test "x$withval" = "xno"; then
MHW_HAVE_PORTAUDIO=no
fi
)

AC_CHECK_HEADER(portaudio.h,mhw_found_header=yes,mhw_found_header=no)
if test "x$mhw_found_header" = "xno"; then
MHW_HAVE_PORTAUDIO=no
fi

if test "x$MHW_HAVE_PORTAUDIO" = "xyes"; then
AC_CHECK_LIB(portaudio,Pa_Initialize,MHW_HAVE_PORTAUDIO=yes,MHW_HAVE_PORTAUDIO=no)
fi

if test "x$MHW_HAVE_PORTAUDIO" = "xyes"; then
AC_CHECK_TYPE(PaTimestamp,,MHW_HAVE_PORTAUDIO=no,[[#include <portaudio.h>]])
fi

if test "x$MHW_HAVE_PORTAUDIO" = "xyes"; then
	AC_DEFINE(HAVE_PORTAUDIO,1,[Enable PortAudio sound driver])
	PORTAUDIO_LIBS="-lportaudio $PORTAUDIO_LIBS"
fi

AC_SUBST(PORTAUDIO_LIBS)

# Check for SDL

HAVE_SDL='no'
enable_sdl='check'

AC_ARG_WITH([sdl], [  --without-sdl           Do not use SDL library],
	[enable_sdl=$withval])

if test "x$enable_sdl" != "xno"; then
	PKG_CHECK_MODULES([SDL], [sdl >= 1.2.13], [HAVE_SDL=yes],
	[
	PKG_CHECK_MODULES([SDL], [sdl2 >= 2.0], [HAVE_SDL=yes],
		[
		if test "x$enable_sdl" = "xyes"; then
			AC_MSG_ERROR([SDL is requested but was not found])
		fi
		])
	])
fi

if test "x$HAVE_SDL" = "xyes" ; then
	AC_DEFINE([HAVE_SDL],[1],[Enable SDL])
fi

dnl Check for AlsaLib

ALSALIB_LIBS=""

AC_CHECK_LIB(asound,snd_pcm_open,HAVE_ALSALIB=yes,HAVE_ALSALIB=no)

AC_DEFUN([AC_WITHOUT_ALSALIB],[
	if test "x$withval" = "xno" ; then
		HAVE_ALSALIB=no
	fi
])

AC_ARG_WITH(alsalib,[  --without-alsalib       Do not compile ALSA-lib driver],AC_WITHOUT_ALSALIB)

AC_CHECK_HEADER(alsa/asoundlib.h,mhw_found_header=yes,mhw_found_header=no)
if test "x$mhw_found_header" = "xno"; then
   HAVE_ALSALIB="no"
fi

if test "x$HAVE_ALSALIB" = "xyes"; then
	ALSALIB_LIBS="-lasound"
	AC_DEFINE(HAVE_ALSALIB,1,[Enable ALSA sound driver])
fi

AC_SUBST(ALSALIB_LIBS)

dnl Check for OSS

HAVE_OSS=yes

AC_DEFUN([AC_WITHOUT_OSS],[
	if test "x$withval" = "xno" ; then
		HAVE_OSS=no
	fi
])

AC_ARG_WITH(oss,[  --without-oss           Do not compile OSS driver],AC_WITHOUT_OSS)

dnl Check for libossaudio, needed for OSS on OpenBSD
AC_CHECK_LIB(ossaudio,_oss_ioctl)

mhw_found_one=no
AC_CHECK_HEADERS(soundcard.h sys/soundcard.h,mhw_found_one=yes)
if test "x$mhw_found_one" = "xno"; then
	HAVE_OSS=no
fi

if test "x$HAVE_OSS" = "xyes"; then
	AC_DEFINE(HAVE_OSS,1,[Enable OSS sound driver])
fi

dnl Check for JACK

MHW_HAVE_JACK=no
MHW_SKIP_JACKTEST=no

AC_DEFUN([AC_WITHOUT_JACK],[
	if test "x$withval" = "xno" ; then
	   MHW_SKIP_JACKTEST='yes'
	fi
])

AC_ARG_WITH(jack,[  --without-jack          Do not compile JACK driver],AC_WITHOUT_JACK)

if test "x$MHW_SKIP_JACKTEST" = "xno"
then
	PKG_CHECK_MODULES(JACK,jack >= 0.98.0,MHW_HAVE_JACK=yes,MHW_HAVE_JACK=no)
fi

if test "x$MHW_HAVE_JACK" = "xyes" ; then
	AC_DEFINE(HAVE_JACK,1,[Enable JACK sound driver])
fi

dnl Check for PulseAudio

MHW_HAVE_PULSE=no
MHW_SKIP_PULSETEST=no

AC_DEFUN([AC_WITHOUT_PULSE],[
	if test "x$withval" = "xno" ; then
	   MHW_SKIP_PULSETEST='yes'
	fi
])

AC_ARG_WITH(pulse,[  --without-pulse          Do not compile PulseAudio driver],AC_WITHOUT_PULSE)

if test "x$MHW_SKIP_PULSETEST" = "xno"
then
	PKG_CHECK_MODULES(PULSE,libpulse >= 0.9.10,MHW_HAVE_PULSE=yes,MHW_HAVE_PULSE=no)
fi

if test "x$MHW_HAVE_PULSE" = "xyes" ; then
	AC_DEFINE(HAVE_PULSEAUDIO,1,[Enable PulseAudio sound driver])
fi

dnl Check for Sun audio

MHW_HAVE_SUNAUDIO=yes

AC_DEFUN([AC_WITHOUT_SUNAUDIO],[
	if test "x$withval" = "xno" ; then
		MHW_HAVE_SUNAUDIO=no
	fi
])

AC_ARG_WITH(sun,[  --without-sun           Do not compile Sun audio driver],AC_WITHOUT_SUNAUDIO)

AC_CHECK_HEADER(sys/audio.h,mhw_found_one=yes,mhw_found_one=no)
if test "x$mhw_found_one" = "xno"; then
	MHW_HAVE_SUNAUDIO=no
fi

if test "x$MHW_HAVE_SUNAUDIO" = "xyes"; then
	AC_DEFINE(HAVE_SUN,1,[Enable Sun audio sound driver])
fi

# Check for large file support

AC_SYS_LARGEFILE
AC_DEFINE(_LARGEFILE_SOURCE,1,[Defined to get fseeko/ftello prototypes])
AC_CHECK_SIZEOF(off_t)

dnl Checks for header files.
AC_CHECK_HEADERS([sys/time.h sys/ioctl.h sys/types.h fcntl.h sched.h libintl.h locale.h])

dnl Ladspa
HAVE_LADSPA=yes
HAVE_LADSPA_GLOBAL=yes
AC_CHECK_HEADER(ladspa.h,,HAVE_LADSPA_GLOBAL=no)
AC_ARG_ENABLE(ladspa,
  [  --disable-ladspa        Do not include any Ladspa support],
  [if test "x$enableval" = "xno"; then HAVE_LADSPA=no; fi])
if test "x$HAVE_LADSPA" = "xyes"; then
	AC_CHECK_HEADER(dlfcn.h,,HAVE_LADSPA=no)
fi
if test "x$HAVE_LADSPA" = "xyes"; then
	AC_CHECK_LIB(dl,dlopen,LIBS="-ldl $LIBS",HAVE_LADSPA=no)
fi
if test "x$HAVE_LADSPA" = "xyes"; then
	AC_DEFINE(HAVE_LADSPA,1,[Enable LADSPA support])
fi
if test "x$HAVE_LADSPA_GLOBAL" = "xyes"; then
        AC_DEFINE(HAVE_SYS_LADSPA_H,1,[Use system LADSPA header])
fi

dnl Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_OFF_T
AC_TYPE_PID_T

dnl Checks for library functions.
AC_CHECK_FUNCS([gettimeofday sched_yield ceill setlocale fseeko ftello lrint lrintf dup2 pow select sqrt strtoul]) 
AC_FUNC_FORK

# Turn on all/most gcc warnings
case "$CC" in
  gcc*) CPPFLAGS="$CPPFLAGS -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers" ;;
  *) ;;
esac

if test x"$enable_gtk_warnings" = x"yes"; then
	CPPFLAGS="$CPPFLAGS -DGDK_DISABLE_DEPRECATED -DGDK_PIXBUF_DISABLE_DEPRECATED -DGDK_PIXBUF_DISABLE_SINGLE_INCLUDES -DGTK_DISABLE_DEPRECATED -DGTK_DISABLE_SINGLE_INCLUDES"
fi


dnl Write output
AC_CONFIG_FILES([Makefile src/Makefile po/Makefile share/Makefile ])
AC_OUTPUT

dnl Display configuration

echo
echo
echo "------------------------"
echo "gWaveEdit configuration"
echo "------------------------"
echo
echo "Version: $VERSION"
echo 
echo "Libraries: "
echo -n "  GTK+ version:           "
if test "x$enable_gtk3" = "xyes" ; then
	echo `$PKG_CONFIG --modversion gtk+-3.0`
else
	echo `$PKG_CONFIG --modversion gtk+-2.0`
fi
echo    "  Libsndfile:             $HAVE_SNDFILE"
echo    "  Libsamplerate:          $HAVE_LIBSAMPLERATE"
echo
echo    "Sound drivers:"
INCLUDED_DRIVERS=""
EXCLUDED_DRIVERS=""
AC_DEFUN([CHECKDRV], [ if test "x$1" = "xyes"; then INCLUDED_DRIVERS="$INCLUDED_DRIVERS $2"; else EXCLUDED_DRIVERS="$EXCLUDED_DRIVERS $2"; fi; ])
CHECKDRV([$HAVE_OSS],[OSS])
CHECKDRV([$HAVE_ALSALIB],[ALSA])
CHECKDRV([$MHW_HAVE_JACK],[JACK])
CHECKDRV([$MHW_HAVE_SUNAUDIO],[Sun])
CHECKDRV([$MHW_HAVE_PORTAUDIO],[PortAudio])
CHECKDRV([$HAVE_SDL],[SDL])
CHECKDRV([$MHW_HAVE_PULSE],[PulseAudio])
echo    "  Included drivers:      $INCLUDED_DRIVERS"
echo    "  Drivers not included:  $EXCLUDED_DRIVERS"
echo -n "  Default sound driver:   $DEFAULT_DRIVER "
if test "x$DEFAULT_DRIVER" = "xauto"; then
echo -n "(detect at startup)"
fi
echo
echo
echo    "Other settings:"


echo    "  LADSPA Support:         $HAVE_LADSPA"
if test "x$HAVE_LADSPA" = "xyes"; then
echo -n "    ladspa.h header file: "
if test "x$HAVE_LADSPA_GLOBAL" = "xyes"; then
echo "system"
else
echo "local"
fi
echo    "    Default plugin path:  $DEFAULT_LADSPA_PATH"
fi

echo -n "  Floating-point samples: "
if test "x$MHW_DOUBLE" = "xyes"; then
	echo "double precision"
else
	echo "normal precision"
fi

echo    "  Default mixer app:      $DEFAULT_MIXERAPP"
echo
echo "Build settings:"
echo -n "  Compiler options:       "
echo    $CFLAGS
echo
echo    "Type make (or gmake) to compile gWaveEdit!"
