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
#include <QTimer>
#include <QSignalMapper>

/* Boost includes */
#include <boost/thread.hpp>

/* Custom classes' includes. */
#include "ui_mainapp.h"
#include "parametric_eq.hh"

/* Default value for Freq and BW. */
#define DEFAULT_FREQ 100
#define DEFAULT_BW 100

/* Min and max for knob values. */
#define KNOB_SET 6
#define KNOB_MIN 0
#define KNOB_MAX 22000

/* How much change for each twist in knob. */
#define KNOB_STEP 10

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

    void on_threadsBlockBox_editingFinished();

    void on_blockNum_editingFinished();

    void setNewDuration(int newDuration);

    void updatePosition();

    void twistKnob1(int value);

    void twistKnob2(int value);

    void twistKnob3(int value);

    void twistKnob4(int value);

    void twistKnob5(int value);

    void twistKnob6(int value);

    void twistKnob7(int value);

    void twistKnob8(int value);

    void twistKnob9(int value);

    void twistKnob10(int value);

    void twistKnob11(int value);

    void twistKnob12(int value);
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

    // Timer to query the samples played by the
    // equalizer.
    QTimer *timer;

    int threadNumPerBlock = 512;
    int maxNumBlock = 200;

    int dialValue[KNOB_SET * 2];
    int previousValue[KNOB_SET * 2];

    void initWindow();
    QString calculateTimeString(int time);
    void setTimeString();
    int knobDirection(int knobNum, int v);
    void setKnobLabel(int knobNum, int direction);
    void freeFilterProperties();
    void initiateProcessing();
};

#endif // MAINAPP_HH
