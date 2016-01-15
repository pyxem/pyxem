dnl Process this file with autoconf to produce a configure script.

AC_INIT(dtr, 1.0.7, taw27@cam.ac.uk)
VERSION=AC_PACKAGE_VERSION

AM_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE(dtr, "$VERSION")

AC_PROG_CC
AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_LN_S

AC_HEADER_STDC
AC_CHECK_HEADERS([fcntl.h stdlib.h string.h unistd.h])
AC_C_CONST
AC_FUNC_MALLOC
AC_FUNC_FORK
AC_HEADER_SYS_WAIT
AC_TYPE_SIGNAL
AC_TYPE_PID_T
AC_CHECK_FUNCS([strdup])
AC_CHECK_FUNCS([floor])
AC_CHECK_FUNCS([pow])
AC_CHECK_FUNCS([rint])
AC_CHECK_FUNCS([sqrt])

AM_PATH_GTK_2_0(2.0.0,,AC_MSG_ERROR([
*** GTK+ 2.6.0 or above is required by DTR but no GTK+2 was found.
*** Please make sure you have the GTK+
*** development files installed. The latest version of GTK+ is
*** always available at http://www.gtk.org/.]))

AC_MSG_CHECKING([GTK version])
if $PKG_CONFIG --atleast-version 2.10.0 gtk+-2.0 ; then
	AC_DEFINE([HAVE_GTK_TEN], [1], [Define to 1 if GTK+ is version 2.10.0 or higher])
	AC_MSG_RESULT([2.10.0 or higher - good])
else
	AC_MSG_RESULT([Lower than 2.10.0 - might be usable if newer than 2.6.0 - let's see...])
	AM_PATH_GTK_2_0(2.6.0,,AC_MSG_ERROR([
*** GTK+2 was found but the version number was too low.
*** GTK+ 2.6.0 or above is required by DTR.  Please make sure you have the GTK+
*** development files installed. The latest version of GTK+ is
*** always available at http://www.gtk.org/.]))
fi

AC_CHECK_LIB(m, main)

AC_CHECK_LIB(gslcblas,main, [],  AC_MSG_ERROR([
       *** gslcblas not found.  gslcblas is required to build Synth2D.]))

AC_CHECK_LIB(gsl, main, [],  AC_MSG_ERROR([
       *** gsl not found.  gsl is required to build Synth2D.]))

AC_MSG_CHECKING([libpng])
if $PKG_CONFIG --atleast-version 1.2.0 libpng ; then
        LIBPNG_VERSION=`$PKG_CONFIG --modversion libpng`
        AC_MSG_RESULT($LIBPNG_VERSION)
        LIBPNG_CFLAGS=`$PKG_CONFIG --cflags libpng`
        LIBPNG_LIBS=`$PKG_CONFIG --libs libpng`
else
        AC_MSG_RESULT([Lower than 1.2.0 or not found])
        AC_MSG_ERROR([
*** libPNG is required to build ugliquant])
fi

AM_PATH_GTKGLEXT_1_0(1.0.0,[echo "$GTKGLEXT_LIBS"],AC_MSG_ERROR([
 	*** gtkglext is required to build dtr; please make sure you have the
 	*** gtkglext development headers installed.]))

AC_CHECK_LIB(GLEW, main, [GLEW_LIBS=-lGLEW],  AC_MSG_ERROR([
*** GLEW not found.  Do you have 'libglew-dev' or similar installed?]))

AC_MSG_CHECKING([Cairo version 1.2.0])
if $PKG_CONFIG --atleast-version 1.2.0 cairo ; then
	CAIRO_VERSION=`$PKG_CONFIG --modversion cairo`
	CAIRO_CFLAGS=`$PKG_CONFIG --cflags cairo`
	CAIRO_LIBS=`$PKG_CONFIG --libs cairo`
	AC_DEFINE([HAVE_CAIRO], [1], [Define to 1 if Cairo is available])
	AC_MSG_RESULT($CAIRO_VERSION)
else
	AC_MSG_RESULT([not found.  Some functions will be unavailable.])
fi

CFLAGS="$CFLAGS $CAIRO_CFLAGS $LIBPNG_CLAGS"
LIBS="$LIBS $CAIRO_LIBS $LIBPNG_LIBS $GLEW_LIBS"

AC_OUTPUT(Makefile src/Makefile data/Makefile)

