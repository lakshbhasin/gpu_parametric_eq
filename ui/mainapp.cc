#include "mainapp.hh"

MainApp::MainApp(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainApp)
{
    // Initialize the ParametricEQ. This first requires initializing
    // NUM_FILTERS filters. We'll assume band-boost filters for now.

    // The first filter will initially be at 64 Hz.
    float freq1 = FREQ_DEFAULT1;           // Hz
    float bandwidth1 = BW_DEFAULT1;        // Hz
    float gain1 = GAIN_DEFAULT1;           // dB (must be positive)
    BandBoostCutProp *bandBCProp1 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp1->omegaNought = 2.0 * M_PI * freq1;
    bandBCProp1->Q = freq1 / bandwidth1;
    bandBCProp1->K = std::pow(10.0, gain1 / 20.0);

    filters[0].type = FT_BAND_BOOST;
    filters[0].bandBCProp = bandBCProp1;

    // The second filter will initially be at 128 Hz.
    float freq2 = FREQ_DEFAULT2;
    float bandwidth2 = BW_DEFAULT2;
    float gain2 = GAIN_DEFAULT2;
    BandBoostCutProp *bandBCProp2 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp2->omegaNought = 2.0 * M_PI * freq2;
    bandBCProp2->Q = freq2 / bandwidth2;
    bandBCProp2->K = std::pow(10.0, gain2 / 20.0);

    filters[1].type = FT_BAND_BOOST;
    filters[1].bandBCProp = bandBCProp2;
    
    // The third filter will initially be at 256 Hz.
    // TODO: change this back so it has no gain.
    float freq3 = FREQ_DEFAULT3;
    float bandwidth3 = BW_DEFAULT3;
    float gain3 = GAIN_DEFAULT3;
    BandBoostCutProp *bandBCProp3 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp3->omegaNought = 2.0 * M_PI * freq3;
    bandBCProp3->Q = freq3 / bandwidth3;
    bandBCProp3->K = std::pow(10.0, gain3 / 20.0);

    filters[2].type = FT_BAND_BOOST;
    filters[2].bandBCProp = bandBCProp3;

    // The fourth filter will initially be at 512 Hz.
    float freq4 = FREQ_DEFAULT4;
    float bandwidth4 = BW_DEFAULT4;
    float gain4 = GAIN_DEFAULT4;
    BandBoostCutProp *bandBCProp4 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp4->omegaNought = 2.0 * M_PI * freq4;
    bandBCProp4->Q = freq4 / bandwidth4;
    bandBCProp4->K = std::pow(10.0, gain4 / 20.0);

    filters[3].type = FT_BAND_BOOST;
    filters[3].bandBCProp = bandBCProp4;

    // The fifth filter will initially be at 1024 Hz.
    float freq5 = FREQ_DEFAULT5;
    float bandwidth5 = BW_DEFAULT5;
    float gain5 = GAIN_DEFAULT5;
    BandBoostCutProp *bandBCProp5 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp5->omegaNought = 2.0 * M_PI * freq5;
    bandBCProp5->Q = freq5 / bandwidth5;
    bandBCProp5->K = std::pow(10.0, gain5 / 20.0);

    filters[4].type = FT_BAND_BOOST;
    filters[4].bandBCProp = bandBCProp5;

    // The sixth filter will initially be at 2048 Hz.
    float freq6 = FREQ_DEFAULT6;
    float bandwidth6 = BW_DEFAULT6;
    float gain6 = GAIN_DEFAULT6;
    BandBoostCutProp *bandBCProp6 = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp6->omegaNought = 2.0 * M_PI * freq6;
    bandBCProp6->Q = freq6 / bandwidth6;
    bandBCProp6->K = std::pow(10.0, gain6 / 20.0);

    filters[5].type = FT_BAND_BOOST;
    filters[5].bandBCProp = bandBCProp6;

    // Construct the parametric EQ. 
    paramEQ = new ParametricEQ(NUM_FILTERS, filters);

    ui->setupUi(this);

    // Init front-end stuff in initWindow()
    initWindow();
}

void MainApp::initBoundDial(QDial *currDial, int idx)
{
    currDial->setValue(dialValue[idx]);
    currDial->setMinimum(KNOB_MIN);
    currDial->setMaximum(KNOB_MAX);
    currDial->setWrapping(false);
}

void MainApp::initDials()
{
    // Set default for freq and bandwidth
    ui->lcdNumber_7->display((int)FREQ_DEFAULT1);
    ui->lcdNumber_8->display((int)FREQ_DEFAULT2);
    ui->lcdNumber_9->display((int)FREQ_DEFAULT3);
    ui->lcdNumber_10->display((int)FREQ_DEFAULT4);
    ui->lcdNumber_11->display((int)FREQ_DEFAULT5);
    ui->lcdNumber_12->display((int)FREQ_DEFAULT6);

    ui->lcdNumber_13->display((int)BW_DEFAULT1);
    ui->lcdNumber_14->display((int)BW_DEFAULT2);
    ui->lcdNumber_15->display((int)BW_DEFAULT3);
    ui->lcdNumber_16->display((int)BW_DEFAULT4);
    ui->lcdNumber_17->display((int)BW_DEFAULT5);
    ui->lcdNumber_18->display((int)BW_DEFAULT6);

    dialValue[0] = (int)FREQ_DEFAULT1;
    dialValue[1] = (int)FREQ_DEFAULT2;
    dialValue[2] = (int)FREQ_DEFAULT3;
    dialValue[3] = (int)FREQ_DEFAULT4;
    dialValue[4] = (int)FREQ_DEFAULT5;
    dialValue[5] = (int)FREQ_DEFAULT6;
    dialValue[6] = (int)BW_DEFAULT1;
    dialValue[7] = (int)BW_DEFAULT2;
    dialValue[8] = (int)BW_DEFAULT3;
    dialValue[9] = (int)BW_DEFAULT4;
    dialValue[10] = (int)BW_DEFAULT5;
    dialValue[11] = (int)BW_DEFAULT6;

    // Set dial properties
    initBoundDial(ui->dial, 0);
    initBoundDial(ui->dial_2, 1);
    initBoundDial(ui->dial_3, 2);
    initBoundDial(ui->dial_4, 3);
    initBoundDial(ui->dial_5, 4);
    initBoundDial(ui->dial_6, 5);
    initBoundDial(ui->dial_7, 6);
    initBoundDial(ui->dial_8, 7);
    initBoundDial(ui->dial_9, 8);
    initBoundDial(ui->dial_10, 9);
    initBoundDial(ui->dial_11, 10);
    initBoundDial(ui->dial_12, 11);

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

void MainApp::initPlot()
{
    ui->customPlot->addGraph();
    ui->customPlot->graph(0)->setPen(QPen(Qt::blue));
    ui->customPlot->graph(0)->setBrush(QBrush(QColor(240, 255, 200)));
    ui->customPlot->graph(0)->setAntialiasedFill(false);
    ui->customPlot->addGraph();
    ui->customPlot->graph(1)->setPen(QPen(Qt::blue));
    ui->customPlot->graph(1)->setLineStyle(QCPGraph::lsNone);
    ui->customPlot->graph(1)->setScatterStyle(QCPScatterStyle::ssDisc);

    ui->customPlot->xAxis->setTickLabelType(QCPAxis::ltDateTime);
    ui->customPlot->xAxis->setDateTimeFormat("mm:ss");
    ui->customPlot->xAxis->setAutoTickStep(false);
    ui->customPlot->xAxis->setTickStep(2);
    ui->customPlot->axisRect()->setupFullAxesBox();

    connect(ui->customPlot->xAxis, SIGNAL(rangeChanged(QCPRange)),
        ui->customPlot->xAxis2, SLOT(setRange(QCPRange)));
    connect(ui->customPlot->yAxis, SIGNAL(rangeChanged(QCPRange)),
        ui->customPlot->yAxis2, SLOT(setRange(QCPRange)));
}

void MainApp::realtimeDataSlot()
{
    double timeValue = duration - elapseTimer->remainingTime() / 1000.0;
    static double lastPointKey = 0;
    if (timeValue - lastPointKey > 0.01)
    {
        // TODO: Superposition of gain values.
        double value0 = qSin((float)gain[0]) * 10.0;
        ui->customPlot->graph(0)->addData(timeValue, value0);
        ui->customPlot->graph(1)->clearData();
        ui->customPlot->graph(1)->addData(timeValue, value0);
        ui->customPlot->graph(0)->removeDataBefore(timeValue - 8);
        ui->customPlot->graph(0)->rescaleValueAxis();
        //ui->customPlot->yAxis->setRange(0.1, 10);
        lastPointKey = timeValue;
    }
    ui->customPlot->xAxis->setRange(timeValue + 0.25, 8, Qt::AlignRight);
    ui->customPlot->replot();
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

    // Num samples, threads / block and max block adjustables
    ui->numSampleBox->setMinimum(512);
    ui->numSampleBox->setMaximum(65536);
    ui->numSampleBox->setValue(numSamples);

    ui->threadsBlockBox->setMinimum(32);
    ui->threadsBlockBox->setMaximum(1024);
    ui->threadsBlockBox->setValue(threadNumPerBlock);

    ui->blockNum->setMinimum(1);
    ui->blockNum->setMaximum(400);
    ui->blockNum->setValue(maxNumBlock);
    
    initDials();

    // Set connection for slider and display
    connect(ui->verticalSlider, SIGNAL(valueChanged(int)),
        this, SLOT(sliderGain1(int)));
    connect(ui->verticalSlider_2, SIGNAL(valueChanged(int)),
        this, SLOT(sliderGain2(int)));
    connect(ui->verticalSlider_3, SIGNAL(valueChanged(int)),
        this, SLOT(sliderGain3(int)));
    connect(ui->verticalSlider_4, SIGNAL(valueChanged(int)),
        this, SLOT(sliderGain4(int)));
    connect(ui->verticalSlider_5, SIGNAL(valueChanged(int)),
        this, SLOT(sliderGain5(int)));
    connect(ui->verticalSlider_6, SIGNAL(valueChanged(int)),
        this, SLOT(sliderGain6(int)));

    // Set default for gain
    ui->verticalSlider->setValue((int)GAIN_DEFAULT1);
    ui->verticalSlider_2->setValue((int)GAIN_DEFAULT2);
    ui->verticalSlider_3->setValue((int)GAIN_DEFAULT3);
    ui->verticalSlider_4->setValue((int)GAIN_DEFAULT4);
    ui->verticalSlider_5->setValue((int)GAIN_DEFAULT5);
    ui->verticalSlider_6->setValue((int)GAIN_DEFAULT6);

    gain[0] = (int)GAIN_DEFAULT1;
    gain[1] = (int)GAIN_DEFAULT2;
    gain[2] = (int)GAIN_DEFAULT3;
    gain[3] = (int)GAIN_DEFAULT4;
    gain[4] = (int)GAIN_DEFAULT5;
    gain[5] = (int)GAIN_DEFAULT6;

    initPlot();

    initDeviceMeta();
}

void MainApp::initDeviceMeta()
{
    // Find device name
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0)
    {
        ui->statusBar->showMessage("No CUDA-able device!", 5000);
        return;
    }
    QString deviceName;
    cudaDeviceProp prop;
    for (int i = 0; i < nDevices; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        deviceName = prop.name;
        break;
    }
    deviceName = "NVidia " + deviceName;

    // Find architecture type
    QString arch;
    int majorCap = prop.major;
    if (majorCap < 2)
        arch = "Tesla";
    else if (majorCap >= 2 && majorCap < 3)
        arch = "Fermi";
    else
        arch = "Kepler";

    arch = "Architecture: " + arch;

    // Get CC
    QString cc = QString::number(prop.major) + "." +
        QString::number(prop.minor);
    cc = "CC: " + cc;

    // Get CUDA version
    int versionInt;
    cudaDriverGetVersion(&versionInt);
    double cudaVersion = (double)versionInt / 1000.0;
    QString version = "CUDA Version: " + QString::number(cudaVersion);

    QString metaInfo = deviceName + "\n" + arch + "\n" + cc +
        "\n" + version;

    ui->textBrowser->setText(metaInfo);
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
 * Set knob's LCD number value based on the direction that a knob turned.
 * Also, update the set of filters.
 */
void MainApp::setKnobValue(int knobNum, int val)
{
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

    dialValue[knobNum] = val;
    currLCD->display(dialValue[knobNum]);

    int filterNum;

    if(knobNum >= NUM_FILTERS)
    {
        // Knob is for changing BW
        filterNum = knobNum - NUM_FILTERS;
    }
    else
    {
        // Knob is for changing freq
        filterNum = knobNum;
    }

    assert(filterNum >= 0 && filterNum < NUM_FILTERS);

    // Note: filter type is unchanged if we're twisting knobs.
    updateFilter(filterNum,                             /* filterNum */ 
                 gain[filterNum],                       /* newGain */ 
                 dialValue[filterNum],                  /* newFreq */
                 dialValue[filterNum + NUM_FILTERS],    /* newBW */
                 filters[filterNum].type                /* filtType */
                 );
}


/**
 * Set gain-related slider's LCD number based on the new slider value.
 * Also, update the gain of the specified filter.
 */
void MainApp::setGainValue(int filterNum, int val)
{
    // Don't do anything if the value is out of bounds.
    if ( !(val <= GAIN_MAX && val >= GAIN_MIN) )
        return;

    assert(filterNum >= 0 && filterNum < NUM_FILTERS);

    // Set the new filter type based on the gain and the old filter type.
    FilterType newFilterType;

    switch(filters[filterNum].type)
    {
        case FT_BAND_BOOST:
        case FT_BAND_CUT:
            newFilterType = FT_BAND_BOOST;

            // Change the filter to a cut if the new gain is < 0
            if (val < 0)
            {
                newFilterType = FT_BAND_CUT;
            }
        
            break;

        default:
            throw std::invalid_argument("Invalid filter type: " +
                    std::to_string(filters[filterNum].type));
    }
    
    // Update gain array and UI.
    gain[filterNum] = val;

    QLCDNumber *currLCD;
    switch(filterNum)
    {
        case 0:
            currLCD = ui->lcdNumber_1;
            break;
        case 1:
            currLCD = ui->lcdNumber_2;
            break;
        case 2:
            currLCD = ui->lcdNumber_3;
            break;
        case 3:
            currLCD = ui->lcdNumber_4;
            break;
        case 4:
            currLCD = ui->lcdNumber_5;
            break;
        case 5:
            currLCD = ui->lcdNumber_6;
            break;
        
        default:
            throw std::invalid_argument("Invalid gain LCD number: " +
                    std::to_string(filterNum));
    }

    currLCD->display(val);
   
    // Update back-end filters.
    updateFilter(filterNum,                             /* filterNum */ 
                 gain[filterNum],                       /* newGain */ 
                 dialValue[filterNum],                  /* newFreq */
                 dialValue[filterNum + NUM_FILTERS],    /* newBW */
                 newFilterType                          /* filtType */
                 );
}


/**
 * A series of functions for each gain-related slider's connection. Note
 * that each slider value is multiplied by NUM_FILTERS since we are taking
 * a weighted average of each individual transfer function, in order to get the final transfer
 * function.
 */
void MainApp::sliderGain1(int value)
{
    setGainValue(0, value);
}


void MainApp::sliderGain2(int value)
{
    setGainValue(1, value);
}


void MainApp::sliderGain3(int value)
{
    setGainValue(2, value);
}


void MainApp::sliderGain4(int value)
{
    setGainValue(3, value);
}


void MainApp::sliderGain5(int value)
{
    setGainValue(4, value);
}


void MainApp::sliderGain6(int value)
{
    setGainValue(5, value);
}


/**
 * A series of functions for each knob's connection.
 */
void MainApp::twistKnob1(int value)
{
    setKnobValue(0, value);
}

void MainApp::twistKnob2(int value)
{
    setKnobValue(1, value);
}

void MainApp::twistKnob3(int value)
{
    setKnobValue(2, value);
}

void MainApp::twistKnob4(int value)
{
    setKnobValue(3, value);
}

void MainApp::twistKnob5(int value)
{
    setKnobValue(4, value);
}

void MainApp::twistKnob6(int value)
{
    setKnobValue(5, value);
}

void MainApp::twistKnob7(int value)
{
    setKnobValue(6, value);
}

void MainApp::twistKnob8(int value)
{
    setKnobValue(7, value);
}

void MainApp::twistKnob9(int value)
{
    setKnobValue(8, value);
}

void MainApp::twistKnob10(int value)
{
    setKnobValue(9, value);
}

void MainApp::twistKnob11(int value)
{
    setKnobValue(10, value);
}

void MainApp::twistKnob12(int value)
{
    setKnobValue(11, value);
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
        plotTimer->stop();
        elapseTimer->stop();
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

        // Start plotting
        plotTimer = new QTimer(this);
        elapseTimer = new QTimer(this);
        elapseTimer->start(duration * 1000.0);
        connect(plotTimer, SIGNAL(timeout()), this, SLOT(realtimeDataSlot()));
        plotTimer->start(0);
    }
    
}

void MainApp::on_numSampleBox_editingFinished()
{
    int newNumSamples = ui->numSampleBox->value();
    try
    {
        paramEQ->setNumBufSamples(newNumSamples);
    }
    catch (std::logic_error e)
    {
        ui->statusBar->showMessage(e.what(), 5000);
        cout << e.what() << endl;
        ui->threadsBlockBox->setValue(numSamples);
        return;
    }
    numSamples = newNumSamples;
    QString msg = "The number of samples to use per buffer (per "
        "channel) has been set to " +  newNumSamples;
    cout << msg.toLocal8Bit().data() << endl;
    ui->statusBar->showMessage(msg, 5000);
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
        ui->threadsBlockBox->setValue(threadNumPerBlock);
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
 * This helper function loads the current filter value and update
 * it accordingly.
 */
void MainApp::updateFilter(int filterNum, float newGain, float newFreq,
                           float newBW, FilterType filtType)
{
    float freq = newFreq;                   // Hz
    float bandwidth = newBW;                // Hz
    float gain = std::fabs(newGain);        // dB (must be positive)    
    
    Filter *oldFilter = &filters[filterNum];
    
    // Free the old Filter's properties
    switch(oldFilter->type)
    {
        case FT_BAND_BOOST:
        case FT_BAND_CUT:
            free(oldFilter->bandBCProp);
            break;
        
        default:
            throw std::invalid_argument("Invalid old filter type: " +
                    std::to_string(oldFilter->type));
    }
    
    // Set the new Filter's type, and set its properties.    
    oldFilter->type = filtType;

    switch(filtType)
    {
        case FT_BAND_BOOST:
        case FT_BAND_CUT:
        {
            // Set up the new BandBoostCutProp
            BandBoostCutProp *bandBCProp = (BandBoostCutProp *)
                malloc(sizeof(BandBoostCutProp));
            
            bandBCProp->omegaNought = 2.0 * M_PI * freq;
            bandBCProp->Q = freq / bandwidth;
            bandBCProp->K = std::pow(10.0, gain / 20.0);
    
            oldFilter->bandBCProp = bandBCProp;
            break;
        }

        default:
            throw std::invalid_argument("Invalid new filter type: " +
                    std::to_string(filtType));
    }
    
    // Have the ParametricEQ signal an update.
    paramEQ->setFilters(filters);
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


/** Destructor **/
MainApp::~MainApp()
{
    freeFilterProperties(); 

    delete paramEQ;
    paramEQ = NULL;

    delete ui;
    ui = NULL;
}


