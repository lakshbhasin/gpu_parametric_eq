#ifndef MAINAPP_HH
#define MAINAPP_HH

/* Standard includes */
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>

/* Qt includes */
#include <QMainWindow>
#include <QFileDialog>
#include <QFileDialog>
#include <QMessageBox>

/* Boost includes */
#include <boost/thread.hpp>

/* Custom classes' includes. */
#include "ui_mainapp.h"
#include "parametric_eq.hh"

namespace Ui {
class MainApp;
}

class MainApp : public QMainWindow
{
    Q_OBJECT

public:
    // The number of filters to use.
    static constexpr uint16_t NUM_FILTERS = 6;

    explicit MainApp(QWidget *parent = 0);
    ~MainApp();

private slots:
    void on_fileSelectButton_clicked();

    void on_processButton_clicked();

private:

    // The internal ParametricEQ to use. 
    ParametricEQ *paramEQ;

    // The Filters to use for the equalizer.
    Filter *filters;

    // The file name of the song to play
    QString currDataFile;

    // The number of seconds of the (modified) song that has already been
    // played.
    int alreadyPlayed;
    
    // The total duration of the song being played.
    int duration;

    // Whether audio is currently being processed.
    bool processing = false;

    // The Qt UI to set up
    Ui::MainApp *ui;
    
    void initWindow();
    QString calculateTimeString(int time);
    void setTimeString();
    void freeFilterProperties();
    void initiateProcessing();
};

#endif // MAINAPP_HH
