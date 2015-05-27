#include "mainapp.hh"

MainApp::MainApp(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainApp)
{
    // Initialize the ParametricEQ. This first requires initializing
    // NUM_FILTERS filters. We'll assume band-boost filters for now.
    filters = (Filter *) malloc(NUM_FILTERS * sizeof(Filter));

    // The first filter will initially be at 64 Hz.
    float freq1 = 64.0;             // Hz
    float bandwidth1 = 64.0;        // Hz
    float gain1 = 0.0;              // dB (must be positive)
    BandBoostCutProp *bandBCProp1 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp1->omegaNought = 2.0 * M_PI * freq1;
    bandBCProp1->Q = freq1 / bandwidth1;
    bandBCProp1->K = std::pow(10.0, gain1 / 20.0);

    filters[0].type = FT_BAND_BOOST;
    filters[0].bandBCProp = bandBCProp1;

    // The second filter will initially be at 128 Hz.
    float freq2 = 128.0;
    float bandwidth2 = 128.0;
    float gain2 = 0.0;
    BandBoostCutProp *bandBCProp2 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp2->omegaNought = 2.0 * M_PI * freq2;
    bandBCProp2->Q = freq2 / bandwidth2;
    bandBCProp2->K = std::pow(10.0, gain2 / 20.0);

    filters[1].type = FT_BAND_BOOST;
    filters[1].bandBCProp = bandBCProp2;
    
    // The third filter will initially be at 256 Hz.
    // TODO: change this back so it has no gain.
    float freq3 = 256.0;
    float bandwidth3 = 256.0;
    float gain3 = 20.0;
    BandBoostCutProp *bandBCProp3 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp3->omegaNought = 2.0 * M_PI * freq3;
    bandBCProp3->Q = freq3 / bandwidth3;
    bandBCProp3->K = std::pow(10.0, gain3 / 20.0);

    filters[2].type = FT_BAND_BOOST;
    filters[2].bandBCProp = bandBCProp3;

    // The fourth filter will initially be at 512 Hz.
    float freq4 = 512.0;
    float bandwidth4 = 512.0;
    float gain4 = 0.0;
    BandBoostCutProp *bandBCProp4 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp4->omegaNought = 2.0 * M_PI * freq4;
    bandBCProp4->Q = freq4 / bandwidth4;
    bandBCProp4->K = std::pow(10.0, gain4 / 20.0);

    filters[3].type = FT_BAND_BOOST;
    filters[3].bandBCProp = bandBCProp4;

    // The fifth filter will initially be at 1024 Hz.
    float freq5 = 1024.0;
    float bandwidth5 = 1024.0;
    float gain5 = 0.0;
    BandBoostCutProp *bandBCProp5 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp5->omegaNought = 2.0 * M_PI * freq5;
    bandBCProp5->Q = freq5 / bandwidth5;
    bandBCProp5->K = std::pow(10.0, gain5 / 20.0);

    filters[4].type = FT_BAND_BOOST;
    filters[4].bandBCProp = bandBCProp5;

    // The sixth filter will initially be at 2048 Hz.
    float freq6 = 2048.0;
    float bandwidth6 = 2048.0;
    float gain6 = 0.0;
    BandBoostCutProp *bandBCProp6 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp6->omegaNought = 2.0 * M_PI * freq6;
    bandBCProp6->Q = freq6 / bandwidth6;
    bandBCProp6->K = std::pow(10.0, gain6 / 20.0);

    filters[5].type = FT_BAND_BOOST;
    filters[5].bandBCProp = bandBCProp6;

    // TODO: Add seventh filter initially at 4096 Hz.

    // Construct the parametric EQ. 
    paramEQ = new ParametricEQ(NUM_FILTERS, filters);

    ui->setupUi(this);

    // Init front-end stuff in initWindow()
    initWindow();
}


/**
 * This function initializes the window and sets some of its properties.
 */
void MainApp::initWindow()
{
    // Set current audio file
    currDataFile = "";

    // Set window name
    QMainWindow::setWindowTitle("GPU EQ");
    QMainWindow::setFixedWidth(1020);
    QMainWindow::setFixedHeight(560);
    ui->label->setAlignment(Qt::AlignRight);

    // Find gui path for logo
    QString guiPath = QDir::currentPath().mid(
        0, QDir::currentPath().indexOf("gpu_parametric_eq") + 17) + "/img/";

    // Get logo
    QPixmap logo(guiPath + "gpu_logo.gif");
    ui->appLogo->setPixmap(logo);

    // Threads / block and max block adjustables
    ui->threadsBlockBox->setMinimum(32);
    ui->threadsBlockBox->setMaximum(1024);
    ui->threadsBlockBox->setValue(threadNumPerBlock);

    ui->blockNum->setMinimum(1);
    ui->blockNum->setMaximum(400);
    ui->blockNum->setValue(maxNumBlock);

    // Set connection for slider and display
    connect(ui->verticalSlider, SIGNAL(valueChanged(int)),
        ui->lcdNumber, SLOT(display(int)));
    connect(ui->verticalSlider_2, SIGNAL(valueChanged(int)),
        ui->lcdNumber_2, SLOT(display(int)));
    connect(ui->verticalSlider_3, SIGNAL(valueChanged(int)),
        ui->lcdNumber_3, SLOT(display(int)));
    connect(ui->verticalSlider_4, SIGNAL(valueChanged(int)),
        ui->lcdNumber_4, SLOT(display(int)));
    connect(ui->verticalSlider_5, SIGNAL(valueChanged(int)),
        ui->lcdNumber_5, SLOT(display(int)));
    connect(ui->verticalSlider_6, SIGNAL(valueChanged(int)),
        ui->lcdNumber_6, SLOT(display(int)));

    // Set default for freq and bandwidth
    ui->lcdNumber_7->display(DEFAULT_FREQ);
    ui->lcdNumber_8->display(DEFAULT_FREQ);
    ui->lcdNumber_9->display(DEFAULT_FREQ);
    ui->lcdNumber_10->display(DEFAULT_FREQ);
    ui->lcdNumber_11->display(DEFAULT_FREQ);
    ui->lcdNumber_12->display(DEFAULT_FREQ);

    ui->lcdNumber_13->display(DEFAULT_BW);
    ui->lcdNumber_14->display(DEFAULT_BW);
    ui->lcdNumber_15->display(DEFAULT_BW);
    ui->lcdNumber_16->display(DEFAULT_BW);
    ui->lcdNumber_17->display(DEFAULT_BW);
    ui->lcdNumber_18->display(DEFAULT_BW);

    for (int k = 0; k < KNOB_SET * 2; k++)
    {
        dialValue[k] = DEFAULT_FREQ;
        previousValue[k] = 0;
        // TODO: set up actual backend value.
    }

    // Set connection for freq and bandwidth
    connect(ui->dial, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob1(int)));
    connect(ui->dial_2, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob2(int)));
    connect(ui->dial_3, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob3(int)));
    connect(ui->dial_4, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob4(int)));
    connect(ui->dial_5, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob5(int)));
    connect(ui->dial_6, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob6(int)));
    connect(ui->dial_7, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob7(int)));
    connect(ui->dial_8, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob8(int)));
    connect(ui->dial_9, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob9(int)));
    connect(ui->dial_10, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob10(int)));
    connect(ui->dial_11, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob11(int)));
    connect(ui->dial_12, SIGNAL(sliderMoved(int)), this,
        SLOT(twistKnob12(int)));
}

/**
 * This helper function calculates a time string from a time (which
 * represents a song's length) in seconds.
 */
QString MainApp::calculateTimeString(int time)
{
    int seconds, hours, minutes;
    seconds = time % 60;
    minutes = time / 60 % 60;
    hours = time / (60 * 60);

    QString timeStr = "";
    if (hours != 0)
    {
        if (hours < 10)
            timeStr += "0" + QString::number(hours);
        else
            timeStr += QString::number(hours);
        timeStr += ":";
    }
    if (minutes > 9)
        timeStr += QString::number(minutes);
    else
    {
        timeStr += "0" + QString::number(minutes);
    }
    timeStr += ":";
    if (seconds > 9)
        timeStr += QString::number(seconds);
    else
    {
        timeStr += "0" + QString::number(seconds);
    }

    return timeStr;
}

/**
 * This function updates the "time" string based on how much of the given
 * song we've played.
 */
void MainApp::setTimeString()
{
    QString played = calculateTimeString(alreadyPlayed);
    QString fullLength = calculateTimeString(duration);
    QString playLabel = played + " / " + fullLength;
    ui->label->setText(playLabel);
    ui->label->repaint();
}

/**
 * This function is called to initialize new duration
 * when a song is loaded after file selection.
 */
void MainApp::setNewDuration(int newDuration)
{
    duration = newDuration;
    alreadyPlayed = 0;
    setTimeString();
    ui->progressBar->setRange(alreadyPlayed, duration);
    ui->progressBar->setValue(alreadyPlayed);
    ui->progressBar->repaint();
}

/**
 * This function is called by timer to query the
 * time played of the current song.
 */
void MainApp::updatePosition()
{
    alreadyPlayed = paramEQ->getPlayedTime();
    setTimeString();
    ui->progressBar->setValue(alreadyPlayed);
    ui->progressBar->repaint();
}


/**
 * This function responds to the "Browse" button being pressed.
 */
void MainApp::on_fileSelectButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
        this, tr("Open WAV file for test"), tr("Audio files (*.wav)"));

    // Check for .wav or .WAV extension
    if ((!filename.isEmpty() &&
        !(filename.contains(".wav") || filename.contains(".WAV"))) )
    {
        QMessageBox::information(this, tr("Error On Selection"),
            "File name \"" + filename + "\" is not a valid WAV file!");
        // Set filename to none
        filename = "";
    }
    else
    {
        ui->textEdit->setText(filename);
        currDataFile = filename;
        char* charDataPath  = currDataFile.toLocal8Bit().data();
        
        // Initialize properties on the Parametric EQ's side.
        // TODO: let number of samples per buffer etc be configurable.
        paramEQ->setSong(charDataPath);
        paramEQ->setNumBufSamples(4096);
        paramEQ->setThreadsPerBlock(threadNumPerBlock);
        paramEQ->setMaxBlocks(maxNumBlock);

        // Read the song's duration and update the time string.
        int newDuration = (paramEQ->getSong())->duration();
        setNewDuration(newDuration);
    }
}


/**
 * This function is called on a separate thread. It begins the parametric
 * equalizer's processing, and does not return until that processing is
 * over.
 */
void MainApp::initiateProcessing()
{
    // Note that the ParametricEQ automatically "stops" itself when it's
    // done processing the entire song.
    paramEQ->startProcessingSound();

    // Once processing is done, change back the "Process" button.
    ui->processButton->setText("Process");
    cout << "Finished processing file: " << currDataFile.toLocal8Bit().data() 
         << endl;
    processing = false;
}

/**
 * Helper function to return the direction turned for a knob.
 */
int MainApp::knobDirection(int knobNum, int v)
{
    int DIRECTION_CLOCKWISE = 1;
    int DIRECTION_ANTICLOCKWISE = -1;
    int difference = previousValue[knobNum] - v;
    int direction = 0;

    // Make sure we have not reached the start.
    if (v == 0)
    {
        if (previousValue[knobNum] == 100)
            direction = DIRECTION_CLOCKWISE;
        else
            direction = DIRECTION_ANTICLOCKWISE;
    }
    else if (v == 100) 
    {
        if (previousValue[knobNum] == 0)
            direction = DIRECTION_ANTICLOCKWISE;
        else
            direction = DIRECTION_CLOCKWISE;
    }
    else
    {
        if (difference > 0)
            direction = DIRECTION_ANTICLOCKWISE;
        else if (difference  < 0)
            direction = DIRECTION_CLOCKWISE;
    }

    // Store the previous value
    previousValue[knobNum] = v;

    return direction;
}

/**
 * Set knob's LCD number value based on direction turned.
 */
void MainApp::setKnobLabel(int knobNum, int direction)
{
    int dial_v = dialValue[knobNum];
    QString msg = "";
    if (dial_v >= KNOB_MAX && direction == 1)
        msg = "Knob cannot change value because maximum value reached.";
    else if (dial_v <= KNOB_MIN && direction == -1)
        msg = "Knob cannot change value because minimum value reached.";

    // Value cannot be changed, set status bar and exit
    if ( !msg.isEmpty() )
    {
        ui->statusBar->showMessage(msg, 5000);
        return;
    }

    // Find the right label to update
    QLCDNumber *currLCD;
    switch(knobNum)
    {
        case 0:
            currLCD = ui->lcdNumber_7;
            break;
        case 1:
            currLCD = ui->lcdNumber_8;
            break;
        case 2:
            currLCD = ui->lcdNumber_9;
            break;
        case 3:
            currLCD = ui->lcdNumber_10;
            break;
        case 4:
            currLCD = ui->lcdNumber_11;
            break;
        case 5:
            currLCD = ui->lcdNumber_12;
            break;
        case 6:
            currLCD = ui->lcdNumber_13;
            break;
        case 7:
            currLCD = ui->lcdNumber_14;
            break;
        case 8:
            currLCD = ui->lcdNumber_15;
            break;
        case 9:
            currLCD = ui->lcdNumber_16;
            break;
        case 10:
            currLCD = ui->lcdNumber_17;
            break;
        case 11:
            currLCD = ui->lcdNumber_18;
            break;
        default:
            currLCD = ui->lcdNumber_7;
    }

    if (direction == 1)
    {
        dialValue[knobNum] += KNOB_STEP;
        // TODO: actually update backend here
        currLCD->display(dialValue[knobNum]);
    }
    else
    {
        dialValue[knobNum] -= KNOB_STEP;
        // TODO: actually update backend here
        currLCD->display(dialValue[knobNum]);
    }
}

/**
 * A series of functions for each knob's connection.
 */
void MainApp::twistKnob1(int value)
{
    int direction = knobDirection(0, value);
    setKnobLabel(0, direction);
}

void MainApp::twistKnob2(int value)
{
    int direction = knobDirection(1, value);
    setKnobLabel(1, direction);
}

void MainApp::twistKnob3(int value)
{
    int direction = knobDirection(2, value);
    setKnobLabel(2, direction);
}

void MainApp::twistKnob4(int value)
{
    int direction = knobDirection(3, value);
    setKnobLabel(3, direction);
}

void MainApp::twistKnob5(int value)
{
    int direction = knobDirection(4, value);
    setKnobLabel(4, direction);
}

void MainApp::twistKnob6(int value)
{
    int direction = knobDirection(5, value);
    setKnobLabel(5, direction);
}

void MainApp::twistKnob7(int value)
{
    int direction = knobDirection(6, value);
    setKnobLabel(6, direction);
}

void MainApp::twistKnob8(int value)
{
    int direction = knobDirection(7, value);
    setKnobLabel(7, direction);
}

void MainApp::twistKnob9(int value)
{
    int direction = knobDirection(8, value);
    setKnobLabel(8, direction);
}

void MainApp::twistKnob10(int value)
{
    int direction = knobDirection(9, value);
    setKnobLabel(9, direction);
}

void MainApp::twistKnob11(int value)
{
    int direction = knobDirection(10, value);
    setKnobLabel(10, direction);
}

void MainApp::twistKnob12(int value)
{
    int direction = knobDirection(11, value);
    setKnobLabel(11, direction);
}

/**
 * This function responds to the "Process"/"Stop" button being pressed.
 */
void MainApp::on_processButton_clicked()
{
    // Check current audio path in text box
    if (currDataFile.isEmpty())
    {
        cout << "Empty file as input!" << endl;
        QMessageBox::information(this, tr("Empty File Input"),
            "Please select a WAV file to process");
        return;
    }

    if (processing)
    {
        // Stop processing.
        ui->processButton->setText("Process");
        cout << "Stopped processing file: " << 
             currDataFile.toLocal8Bit().data() << endl;
        
        processing = false;
        paramEQ->stopProcessingSound();
        timer->stop();
    }
    else
    {
        // Start processing, but on a separate thread.
        ui->processButton->setText("Stop");
        cout << "Processing file: " << currDataFile.toLocal8Bit().data() 
             << endl;
        
        processing = true;
        boost::thread processingThread(boost::bind(
                    &MainApp::initiateProcessing, this));
        timer = new QTimer(this);
        connect(timer, SIGNAL(timeout()), this, SLOT(updatePosition()));
        // Timer to query the samples played every half second.
        timer->start(500);
    }
    
}

void MainApp::on_threadsBlockBox_editingFinished()
{
    int newThreadsBlock = ui->threadsBlockBox->value();
    try
    {
        paramEQ->setThreadsPerBlock(newThreadsBlock);
    }
    catch (std::logic_error e)
    {
        ui->statusBar->showMessage(e.what(), 5000);
        cout << e.what() << endl;
        ui->threadsBlockBox->setValue(200);
        return;
    }
    threadNumPerBlock = newThreadsBlock;
    QString msg = "Threads per block has been set to " + \
        threadNumPerBlock;
    cout << msg.toLocal8Bit().data() << endl;
    ui->statusBar->showMessage(msg, 5000);
}



void MainApp::on_blockNum_editingFinished()
{
    int newBlockNum = ui->blockNum->value();
    try
    {
        paramEQ->setMaxBlocks(newBlockNum);
    }
    catch (std::logic_error e)
    {
        ui->statusBar->showMessage(e.what(), 5000);
        cout << e.what() << endl;
        ui->blockNum->setValue(maxNumBlock);
        return;
    }
    maxNumBlock = newBlockNum;
    QString msg = "Max number of blocks has been set to " + \
        int(maxNumBlock);
    cout << msg.toLocal8Bit().data() << endl;
    ui->statusBar->showMessage(msg, 5000);
}


/**
 * This helper function just frees the filter properties for each filter,
 * which is useful for when filters need to change.
 */
void MainApp::freeFilterProperties()
{
    for (uint16_t i = 0; i < NUM_FILTERS; i++)
    {
        Filter thisFilter = filters[i];

        switch(thisFilter.type)
        {
            case FT_BAND_BOOST:
            case FT_BAND_CUT:
                // Free the BandBoostCutProp.
                free(thisFilter.bandBCProp);
                break;
            
            default:
                throw std::invalid_argument("Invalid filter type: " +
                        std::to_string(thisFilter.type));
        }
    }
}


/* Destructor */
MainApp::~MainApp()
{
    freeFilterProperties(); 
    free(filters);
    filters = NULL;

    delete paramEQ;
    paramEQ = NULL;

    delete ui;
    ui = NULL;
}

