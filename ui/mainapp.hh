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

/* Default value for Freq, BW, and gain. */
#define FREQ_DEFAULT1 64.0
#define FREQ_DEFAULT2 128.0
#define FREQ_DEFAULT3 256.0
#define FREQ_DEFAULT4 512.0
#define FREQ_DEFAULT5 1024.0
#define FREQ_DEFAULT6 2048.0
#define BW_DEFAULT1 64.0
#define BW_DEFAULT2 128.0
#define BW_DEFAULT3 256.0
#define BW_DEFAULT4 512.0
#define BW_DEFAULT5 1024.0
#define BW_DEFAULT6 2048.0
#define GAIN_DEFAULT1 0.0
#define GAIN_DEFAULT2 0.0
#define GAIN_DEFAULT3 20.0
#define GAIN_DEFAULT4 0.0
#define GAIN_DEFAULT5 0.0
#define GAIN_DEFAULT6 0.0

/* How much change for each twist in knob. */
#define KNOB_SET 6
#define KNOB_MIN 10 // Should be > 0
#define KNOB_MAX 22000
#define GAIN_MAX 30
#define GAIN_MIN -30

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

    void sliderGain1(int value);

    void sliderGain2(int value);

    void sliderGain3(int value);

    void sliderGain4(int value);

    void sliderGain5(int value);

    void sliderGain6(int value);

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

    void realtimeDataSlot();

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
    QTimer *dataTimer;

    int threadNumPerBlock = 512;
    int maxNumBlock = 200;

    // Keep track of current freq and bw values
    int dialValue[KNOB_SET * 2];

    // Keep track of current gain values
    int gain[KNOB_SET];

    QString calculateTimeString(int time);
    void setTimeString();
    void initBoundDial(QDial *currDial, int idx);
    void initDials();
    void initPlot();
    void setKnobValue(int knobNum, int direction);
    void initWindow();
    void freeFilterProperties();
    void initiateProcessing();
    void updateFilter(int filterNum, int newGain, int newFreq,
        int newBW, bool cut);
};

#endif // MAINAPP_HH
