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
    ui->threadsBlockBox->setValue(512);
    
    ui->blockNum->setMinimum(1);
    ui->blockNum->setMaximum(400);
    ui->blockNum->setValue(200);
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
    if (minutes > 10)
        timeStr += QString::number(minutes);
    else
    {
        timeStr += "0" + QString::number(minutes);
    }
    timeStr += ":";
    if (seconds > 10)
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
        paramEQ->setThreadsPerBlock(512);
        paramEQ->setMaxBlocks(200);

        // Read the song's duration and update the time string.
        duration = (paramEQ->getSong())->duration();
        alreadyPlayed = 0;
        setTimeString();
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
    }
    
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

