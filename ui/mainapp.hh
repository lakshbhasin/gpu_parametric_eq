#ifndef MAINAPP_HH
#define MAINAPP_HH

/* Standard includes */
#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <complex>
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
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

/* Custom classes' includes. */
#include "ui_mainapp.h"
#include "parametric_eq.hh"

/* Default value for GPU. */
#define NUM_SAMPLES 4096
#define THREADS_PER_BLOCK 512
#define MAX_NUM_BLOCK 200

/* Initial default values for filters. */
#define DEFAULT_FILTER_TYPE1 SFT_LOW_SHELF
#define DEFAULT_FILTER_TYPE2 SFT_BAND
#define DEFAULT_FILTER_TYPE3 SFT_BAND
#define DEFAULT_FILTER_TYPE4 SFT_BAND
#define DEFAULT_FILTER_TYPE5 SFT_BAND
#define DEFAULT_FILTER_TYPE6 SFT_HIGH_SHELF

/* Default value for Freq, BW, and gain. */
#define FREQ_DEFAULT1 64.0
#define FREQ_DEFAULT2 160.0
#define FREQ_DEFAULT3 410.0
#define FREQ_DEFAULT4 680.0
#define FREQ_DEFAULT5 1800.0
#define FREQ_DEFAULT6 4150.0
#define BW_DEFAULT1 24.0
#define BW_DEFAULT2 80.0
#define BW_DEFAULT3 180.0
#define BW_DEFAULT4 320.0
#define BW_DEFAULT5 700.0
#define BW_DEFAULT6 800.0
#define GAIN_DEFAULT1 0.0
#define GAIN_DEFAULT2 0.0
#define GAIN_DEFAULT3 0.0
#define GAIN_DEFAULT4 0.0
#define GAIN_DEFAULT5 0.0
#define GAIN_DEFAULT6 0.0

/* Spacing factors for the frequency axis on the plot. */
#define MIN_FREQ_SPACE_FACTOR   0.95
#define MAX_FREQ_SPACE_FACTOR   1.05


/**
 * Selectable filter types (via the QComboBox). Note that these are not the
 * same as FilterTypes; these are much broader.
 */
typedef enum SelectableFilterType
{
    SFT_BAND,           /* Band boost/cut filters. */
    SFT_HIGH_SHELF,     /* High shelving filter */
    SFT_LOW_SHELF       /* Low shelving filter */
} SelectableFilterType;


namespace Ui
{
    class MainApp;
}


/** 
 * This class re-definition is needed for QComboBox
  * to be redefined for our new CSS
  */
class SelectionKillerDelegate : public QItemDelegate
{
    virtual void paint(QPainter *painter,
        const QStyleOptionViewItem &option,
        const QModelIndex &index) const override
     {
         QStyleOptionViewItem myOption = option;
         myOption.state &= (~QStyle::State_Selected);
         QItemDelegate::paint (painter, myOption, index);
     }
 };



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

    void on_numSampleBox_editingFinished();
    void on_threadsBlockBox_editingFinished();
    void on_blockNum_editingFinished();

    void setNewDuration(float newDuration);

    void songListener();

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

    void showDropdown1();
    void showDropdown2();
    void showDropdown3();
    void showDropdown4();
    void showDropdown5();
    void showDropdown6();
    void selectFilter1(int val);
    void selectFilter2(int val);
    void selectFilter3(int val);
    void selectFilter4(int val);
    void selectFilter5(int val);
    void selectFilter6(int val);

private:

    // The minimum and maximum gains per filter.
    static constexpr float GAIN_MIN = -18.0;        // must be negative!
    static constexpr float GAIN_MAX = 18.0;

    // This is how long (in milliseconds) our "song listener" will wait
    // until it updates various aspects related to the song. 
    static constexpr int LISTENER_UPD_MS = 100;

    // The "resolution" of the progress bar (which tracks how much of the
    // song we've played). This is in units of "steps" per second (i.e.
    // setting this to 10 means that we can resolve song playing to a tenth
    // of a second).
    //
    // It makes sense for this to equal the number of "listener updates" we
    // have per second.
    static constexpr int PROG_BAR_RES_PER_S = (int) (1000.0 / 
                                                     LISTENER_UPD_MS);

    // The internal ParametricEQ to use. 
    ParametricEQ *paramEQ;

    // The Filters to use for the equalizer.
    Filter filters[NUM_FILTERS];

    // The file name of the song to play
    QString currDataFile;

    // The number of seconds of the (modified) song that has already been
    // played.
    float alreadyPlayed;
    
    // The total duration of the song being played, in seconds.
    float duration;

    // Whether audio is currently being processed.
    bool processing = false;

    // Whether the plot has been initialized.
    bool plotInitialized = false;

    // The Qt UI to set up
    Ui::MainApp *ui;

    // Timer to listen to song updates.
    QTimer *songUpdatesTimer = NULL;

    // Timer for data plotting.
    QTimer *plotTimer = NULL;

    // The thread that's calling the ParametricEQ.
    boost::thread *processingThread = NULL;

    // Variables for GPU backend.
    int numSamples = NUM_SAMPLES;
    int threadNumPerBlock = THREADS_PER_BLOCK;
    int maxNumBlock = MAX_NUM_BLOCK;

    // Keep track of current frequency and BW values
    int dialValue[NUM_FILTERS * 2];

    // Keep track of current gain values
    int gain[NUM_FILTERS];

    // Store current filter types. These are not the same as the
    // FilterTypes used internally in parametric_eq_cuda.cu; they're more
    // broad.
    SelectableFilterType selFilterTypes[NUM_FILTERS];

    // Use to convert audio play time to string that makes sense.
    QString calculateTimeString(float timeFloat);
    void setTimeString();

    // Use to connect knob variables frontend to backend.
    void initBoundDial(QCustomDial *currDial, int idx);
    void initDials();
    void setKnobValue(int knobNum, int val);
    void setGainValue(int filterNum, int val);

    // Set up real-time data plotting.
    void initPlot();
    
    void initDeviceMeta();
    void setSongProperties();
    void initWindow();
    
    void setupFilterLogos(QString guiPath);
    QIcon getImageType(SelectableFilterType type, QString guiPath);
    void handleSFTUpdates(int filterNum, QPushButton *buttonToUpdate,
        SelectableFilterType selFiltType);
    
    void freeFilterProperties();

    void initiateProcessing();
    void handleStopProcessing();
    
    // Use to update filters by replacing the one at index "filterNum". 
    void updateFilter(int filterNum, float newGain, float newFreq,
                      float newBW, FilterType filtType);

    void updatePlot();
};

#endif // MAINAPP_HH
