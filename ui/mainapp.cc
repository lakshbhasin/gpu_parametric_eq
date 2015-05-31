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

/**
  * Helper function to return the correct QIcon for a
  * request based on specific filter type.
  */
QIcon MainApp::getImageType(int type, QString guiPath)
{
    QPixmap filterLogo;
    if (type == 0)
    {
        // high sheiving
        filterLogo = QPixmap(guiPath + "rising_big.png");
    }
    else if (type == 1)
    {
        // peak
        filterLogo = QPixmap(guiPath + "peak_big.png");
    }
    else
    {
        // low sheiving
        filterLogo = QPixmap(guiPath + "falling_big.png");
    }
    QIcon filterIcon(filterLogo);

    return filterIcon;
}

/**
  * A serious of SLOT functions to change filter logo and
  * filterType values to store when the dropdown menu is
  * activated.
  */
void MainApp::selectFilter1(int val)
{
    QString guiPath = QDir::currentPath().mid(
        0, QDir::currentPath().indexOf("gpu_parametric_eq") + 17) + "/img/";
    QIcon newFilter = getImageType(val, guiPath);
    filterType[0] = val;
    ui->filter1Logo->setIcon(newFilter);
}

void MainApp::selectFilter2(int val)
{
    QString guiPath = QDir::currentPath().mid(
        0, QDir::currentPath().indexOf("gpu_parametric_eq") + 17) + "/img/";
    QIcon newFilter = getImageType(val, guiPath);
    filterType[1] = val;
    ui->filter2Logo->setIcon(newFilter);
}

void MainApp::selectFilter3(int val)
{
    QString guiPath = QDir::currentPath().mid(
        0, QDir::currentPath().indexOf("gpu_parametric_eq") + 17) + "/img/";
    QIcon newFilter = getImageType(val, guiPath);
    filterType[2] = val;
    ui->filter3Logo->setIcon(newFilter);
}

void MainApp::selectFilter4(int val)
{
    QString guiPath = QDir::currentPath().mid(
        0, QDir::currentPath().indexOf("gpu_parametric_eq") + 17) + "/img/";
    QIcon newFilter = getImageType(val, guiPath);
    filterType[3] = val;
    ui->filter4Logo->setIcon(newFilter);
}

void MainApp::selectFilter5(int val)
{
    QString guiPath = QDir::currentPath().mid(
        0, QDir::currentPath().indexOf("gpu_parametric_eq") + 17) + "/img/";
    QIcon newFilter = getImageType(val, guiPath);
    filterType[4] = val;
    ui->filter5Logo->setIcon(newFilter);
}

void MainApp::selectFilter6(int val)
{
    QString guiPath = QDir::currentPath().mid(
        0, QDir::currentPath().indexOf("gpu_parametric_eq") + 17) + "/img/";
    QIcon newFilter = getImageType(val, guiPath);
    filterType[5] = val;
    ui->filter6Logo->setIcon(newFilter);
}

/**
  * One of the initializing functions to set up all filters at the
  * start of the program and to connect them to back-end SIGNALS and SLOTS.
  */
void MainApp::setupFilterLogos(QString guiPath)
{
    filterType[0] = DEFAULT_FILTER_TYPE1;
    filterType[1] = DEFAULT_FILTER_TYPE2;
    filterType[2] = DEFAULT_FILTER_TYPE3;
    filterType[3] = DEFAULT_FILTER_TYPE4;
    filterType[4] = DEFAULT_FILTER_TYPE5;
    filterType[5] = DEFAULT_FILTER_TYPE6;

    ui->comboBox->setItemDelegate (new SelectionKillerDelegate);
    ui->comboBox_2->setItemDelegate (new SelectionKillerDelegate);
    ui->comboBox_3->setItemDelegate (new SelectionKillerDelegate);
    ui->comboBox_4->setItemDelegate (new SelectionKillerDelegate);
    ui->comboBox_5->setItemDelegate (new SelectionKillerDelegate);
    ui->comboBox_6->setItemDelegate (new SelectionKillerDelegate);

    ui->comboBox->insertItem(0, getImageType(0, guiPath), "High");
    ui->comboBox->insertItem(1, getImageType(1, guiPath), "Band");
    ui->comboBox->insertItem(2, getImageType(2, guiPath), "Low");
    ui->comboBox_2->insertItem(0, getImageType(0, guiPath), "High");
    ui->comboBox_2->insertItem(1, getImageType(1, guiPath), "Band");
    ui->comboBox_2->insertItem(2, getImageType(2, guiPath), "Low");
    ui->comboBox_3->insertItem(0, getImageType(0, guiPath), "High");
    ui->comboBox_3->insertItem(1, getImageType(1, guiPath), "Band");
    ui->comboBox_3->insertItem(2, getImageType(2, guiPath), "Low");
    ui->comboBox_4->insertItem(0, getImageType(0, guiPath), "High");
    ui->comboBox_4->insertItem(1, getImageType(1, guiPath), "Band");
    ui->comboBox_4->insertItem(2, getImageType(2, guiPath), "Low");
    ui->comboBox_5->insertItem(0, getImageType(0, guiPath), "High");
    ui->comboBox_5->insertItem(1, getImageType(1, guiPath), "Band");
    ui->comboBox_5->insertItem(2, getImageType(2, guiPath), "Low");
    ui->comboBox_6->insertItem(0, getImageType(0, guiPath), "High");
    ui->comboBox_6->insertItem(1, getImageType(1, guiPath), "Band");
    ui->comboBox_6->insertItem(2, getImageType(2, guiPath), "Low");

    ui->filter1Logo->setIcon(getImageType(filterType[0], guiPath));
    ui->filter2Logo->setIcon(getImageType(filterType[1], guiPath));
    ui->filter3Logo->setIcon(getImageType(filterType[2], guiPath));
    ui->filter4Logo->setIcon(getImageType(filterType[3], guiPath));
    ui->filter5Logo->setIcon(getImageType(filterType[4], guiPath));
    ui->filter6Logo->setIcon(getImageType(filterType[5], guiPath));

    connect(ui->filter1Logo, SIGNAL(clicked()), this, SLOT(showDropdown1()));
    connect(ui->filter2Logo, SIGNAL(clicked()), this, SLOT(showDropdown2()));
    connect(ui->filter3Logo, SIGNAL(clicked()), this, SLOT(showDropdown3()));
    connect(ui->filter4Logo, SIGNAL(clicked()), this, SLOT(showDropdown4()));
    connect(ui->filter5Logo, SIGNAL(clicked()), this, SLOT(showDropdown5()));
    connect(ui->filter6Logo, SIGNAL(clicked()), this, SLOT(showDropdown6()));

    connect(ui->comboBox, SIGNAL(activated(int)), this, SLOT(selectFilter1(int)));
    connect(ui->comboBox_2, SIGNAL(activated(int)), this, SLOT(selectFilter2(int)));
    connect(ui->comboBox_3, SIGNAL(activated(int)), this, SLOT(selectFilter3(int)));
    connect(ui->comboBox_4, SIGNAL(activated(int)), this, SLOT(selectFilter4(int)));
    connect(ui->comboBox_5, SIGNAL(activated(int)), this, SLOT(selectFilter5(int)));
    connect(ui->comboBox_6, SIGNAL(activated(int)), this, SLOT(selectFilter6(int)));
    //connect(ui->filter1Logo, SIGNAL(clicked()), ui->comboBox, SLOT(showPopup()));
}

void MainApp::initBoundDial(QCustomDial *currDial, int idx)
{
    currDial->setFrequency(dialValue[idx]);
}

void MainApp::initDials()
{
    // Set default values for freq and bandwidth
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
    initBoundDial(ui->freq_dial_1, 0);
    initBoundDial(ui->freq_dial_2, 1);
    initBoundDial(ui->freq_dial_3, 2);
    initBoundDial(ui->freq_dial_4, 3);
    initBoundDial(ui->freq_dial_5, 4);
    initBoundDial(ui->freq_dial_6, 5);
    initBoundDial(ui->bw_dial_1, 6);
    initBoundDial(ui->bw_dial_2, 7);
    initBoundDial(ui->bw_dial_3, 8);
    initBoundDial(ui->bw_dial_4, 9);
    initBoundDial(ui->bw_dial_5, 10);
    initBoundDial(ui->bw_dial_6, 11);

    // Set connection for freq and bandwidth
    connect(ui->freq_dial_1, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob1(int)));
    connect(ui->freq_dial_2, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob2(int)));
    connect(ui->freq_dial_3, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob3(int)));
    connect(ui->freq_dial_4, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob4(int)));
    connect(ui->freq_dial_5, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob5(int)));
    connect(ui->freq_dial_6, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob6(int)));
    connect(ui->bw_dial_1, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob7(int)));
    connect(ui->bw_dial_2, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob8(int)));
    connect(ui->bw_dial_3, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob9(int)));
    connect(ui->bw_dial_4, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob10(int)));
    connect(ui->bw_dial_5, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob11(int)));
    connect(ui->bw_dial_6, SIGNAL(valueChanged(int)), this,
        SLOT(twistKnob12(int)));
}

void MainApp::initPlot()
{
    ui->customPlot->addGraph();

    /* Begin color/font configuration */
    
    // The background and plotting rectangle's colors
    ui->customPlot->setBackground(QColor(13, 29, 39));
    ui->customPlot->axisRect()->setBackground(QColor(13, 29, 39));
    
    // The color of the lines themselves
    ui->customPlot->graph(0)->setPen(QPen(QColor(214, 221, 225)));

    // The color of the fill used for the graph. The alpha makes this
    // fairly transparent.
    ui->customPlot->graph(0)->setBrush(QBrush(QColor(214, 221, 225, 20)));
    ui->customPlot->graph(0)->setAntialiasedFill(false);

    // Add a dummy graph so we can fill down to very negative values. This
    // dummy graph should be plotted at the lowest possible superposition
    // of gains, with some vertical spacing added.
    ui->customPlot->addGraph();

    double minFreq = QCustomDial::minFrequency() * MIN_FREQ_SPACE_FACTOR;
    double maxFreq = QCustomDial::maxFrequency() * MAX_FREQ_SPACE_FACTOR;
    double lowestYVal = GAIN_MIN * NUM_FILTERS - 10;

    ui->customPlot->graph(1)->addData(minFreq, lowestYVal);
    ui->customPlot->graph(1)->addData(maxFreq, lowestYVal);
    
    ui->customPlot->graph(0)->setChannelFillGraph(
            ui->customPlot->graph(1));

    // The color/fonts of the labels and tick labels
    QColor labelColor(101, 120, 133);
    QFont labelFont("Arial", 10);
    QFont tickLabelFont("Arial", 8);

    ui->customPlot->xAxis->setLabelColor(labelColor);
    ui->customPlot->xAxis->setLabelFont(labelFont);
    ui->customPlot->yAxis->setLabelColor(labelColor);
    ui->customPlot->yAxis->setLabelFont(labelFont);
    ui->customPlot->xAxis->setTickLabelColor(labelColor);
    ui->customPlot->xAxis->setTickLabelFont(tickLabelFont);
    ui->customPlot->yAxis->setTickLabelColor(labelColor);
    ui->customPlot->yAxis->setTickLabelFont(tickLabelFont);

    // The color of axis lines, including their ticks and subticks.
    QColor axisColor(62, 80, 93);
    QPen axisPen(axisColor);
    
    ui->customPlot->xAxis->setBasePen(axisPen);
    ui->customPlot->yAxis->setBasePen(axisPen);
    ui->customPlot->xAxis2->setBasePen(axisPen);
    ui->customPlot->yAxis2->setBasePen(axisPen);
    ui->customPlot->xAxis->setTickPen(axisPen);
    ui->customPlot->yAxis->setTickPen(axisPen);
    ui->customPlot->xAxis2->setTickPen(axisPen);
    ui->customPlot->yAxis2->setTickPen(axisPen);
    ui->customPlot->xAxis->setSubTickPen(axisPen);
    ui->customPlot->yAxis->setSubTickPen(axisPen);
    ui->customPlot->xAxis2->setSubTickPen(axisPen);
    ui->customPlot->yAxis2->setSubTickPen(axisPen);

    // The color of subgrid and regular grid lines.
    QColor gridColor(70, 100, 118, 40);     // alpha included for transp.
    QPen gridPen(gridColor);
    
    ui->customPlot->xAxis->grid()->setPen(gridPen);
    ui->customPlot->yAxis->grid()->setPen(gridPen);
    ui->customPlot->xAxis->grid()->setSubGridPen(gridPen);
    ui->customPlot->yAxis->grid()->setSubGridPen(gridPen);
    
    // The color of the zero line
    ui->customPlot->yAxis->grid()->setZeroLinePen(QPen(Qt::transparent));

    /* End color configuration */

    /* Begin axis configuration */

    ui->customPlot->xAxis->setTickLabelType(QCPAxis::ltNumber);
    ui->customPlot->yAxis->setTickLabelType(QCPAxis::ltNumber);

    ui->customPlot->yAxis->grid()->setSubGridVisible(true);
    ui->customPlot->yAxis->setAutoSubTicks(false); 
    ui->customPlot->yAxis->setSubTickCount(1);
    ui->customPlot->xAxis->grid()->setSubGridVisible(true);
    ui->customPlot->yAxis->setAutoSubTicks(false);  
    ui->customPlot->xAxis->setSubTickCount(4);
    
    // Axis labels
    ui->customPlot->xAxis->setLabel("Frequency (Hz)");
    ui->customPlot->yAxis->setLabel("Gain (dB)");

    // Make the x-axis logarithmic
    ui->customPlot->xAxis->setScaleType(QCPAxis::stLogarithmic);
    ui->customPlot->xAxis->setScaleLogBase(QCustomDial::LOG_BASE);

    // Let the x-axis cover the entire possible frequency range (based on
    // the knobs). Leave some padding for labels.
    ui->customPlot->xAxis->setRange(minFreq, maxFreq);

    /* End axis configuration */
    
    // Disable all interactions.
    ui->customPlot->setInteractions(0);

    // Copy over some properties to xAxis2 and yAxis2    
    ui->customPlot->axisRect()->setupFullAxesBox();

    plotInitialized = true;

    // Plot the initial transfer function
    updatePlot();
}

void MainApp::updatePlot()
{
    // Frequency and gain vectors. The frequencies will be logarithmically
    // spaced, with the same spacing as specified in QCustomDial.
    int numPts = QCustomDial::KNOB_MAX - QCustomDial::KNOB_MIN;
    
    QVector<double> freqs(numPts);           // Hz
    QVector<double> gainsDB(numPts);         // dB

    // Keep track of the minimum and maximum gains (in dB). Initialize
    // these to twice GAIN_MIN and GAIN_MAX respectively, so that there's
    // some extra space in our plot in the "normal case".
    double gainsDBMin = GAIN_MIN * 2.0;
    double gainsDBMax = GAIN_MAX * 2.0;

    // For each point, take the "superposition" (i.e. multiplication) of
    // the transfer functions, and then take the absolute value to get the
    // gain.
    for (int i = 0; i < numPts; i++)
    {
        // Frequency = FREQ_MULT * LOG_BASE^(i / EXP_DIV). Don't round in
        // this case.
        int thisFreq = QCustomDial::FREQ_MULT * 
            std::pow(QCustomDial::LOG_BASE, i / QCustomDial::EXP_DIV);

        // For the first and last points, use the endpoints of the graphing
        // region.
        if (i == 0)
        {
            thisFreq *= MIN_FREQ_SPACE_FACTOR;
        }
        else if (i == numPts - 1)
        {
            thisFreq *= MAX_FREQ_SPACE_FACTOR;
        }

        freqs[i] = thisFreq;

        // The Laplace-transform variable for this frequency.
        std::complex<double> s(0.0, 2.0 * M_PI * thisFreq);

        // The complex transfer function value at this frequency (a
        // multiplication of all the individual transfer functions).
        std::complex<double> output(1.0, 0.0);

        for (int filtNum = 0; filtNum < NUM_FILTERS; filtNum ++)
        {
            Filter thisFilter = filters[filtNum];
            FilterType thisFilterType = thisFilter.type;
            
            std::complex<double> sSq;
            double omegaNought, Q, K, omegaNoughtOvQ, omegaNoughtSq;
            
            switch (thisFilterType)
            {
                case FT_BAND_BOOST:
                case FT_BAND_CUT:
                {
                    // For boosts, use the transfer function: 
                    //
                    // H(s) = (s^2 + K * omegaNought/Q * s + omegaNought^2)
                    //        / (s^2 + omegaNought/Q * s + omegaNought^2)
                    // 
                    // And use the reciprocal of this for cuts.
                    
                    omegaNought = (double) thisFilter.bandBCProp->omegaNought;
                    Q = (double) thisFilter.bandBCProp->Q;
                    K = (double) thisFilter.bandBCProp->K;
                    
                    // Do some precomputation
                    sSq = s * s;
                    omegaNoughtOvQ = omegaNought / Q;
                    omegaNoughtSq = omegaNought * omegaNought;
                    
                    // The numerator and denominator of the above H(s) for
                    // boosts.
                    std::complex<double> numerBoost = sSq + 
                        K * omegaNoughtOvQ * s + omegaNoughtSq;
                    std::complex<double> denomBoost = sSq + 
                        omegaNoughtOvQ * s + omegaNoughtSq;
                    
                    // If this is a boost, then just add numerBoost /
                    // denomBoost to the output element. Otherwise, if it's
                    // a cut, add the reciprocal of this.
                    std::complex<double> quot;
                    
                    if (thisFilterType == FT_BAND_BOOST)
                    {
                        quot = numerBoost / denomBoost;
                    }
                    else
                    {
                        quot = denomBoost / numerBoost;
                    }
                    
                    // Multiply the previous transfer function by this one.
                    output *= quot;

                    break;
                }
                
                default:
                    throw std::invalid_argument("Unknown filter type: " + 
                            std::to_string(thisFilterType));
            }
        }

        // Get the gain and store it in dB.
        double thisGain = std::abs(output);
        double thisGainDB = 20.0 * std::log10(thisGain);
        
        gainsDB[i] = thisGainDB;

        gainsDBMin = std::min(thisGainDB, gainsDBMin);
        gainsDBMax = std::max(thisGainDB, gainsDBMax);
    }
    
    // Graph gain as a function of frequency
    ui->customPlot->graph(0)->setData(freqs, gainsDB);
        
    // Set the y-axis scale based on the min and max found earlier (which
    // might extend beyond the data in the "normal" case). Note that we
    // always want the range to be symmetric, and we give a few dB of
    // padding space.
    double halfRange = std::max(std::abs(gainsDBMin),
                                std::abs(gainsDBMax)) + 2.0;

    ui->customPlot->yAxis->setRange(-halfRange, halfRange);
    ui->customPlot->yAxis2->setRange(-halfRange, halfRange);

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
    QMainWindow::setFixedHeight(540);
    ui->label->setAlignment(Qt::AlignRight);

    // Find gui path for logo
    QString guiPath = QDir::currentPath().mid(
        0, QDir::currentPath().indexOf("gpu_parametric_eq") + 17) + "/img/";

    // Get logo
    QPixmap logo(guiPath + "gpuLogo_small_blue.png");
    ui->appLogo->setPixmap(logo);

    // Set up logos for filters
    setupFilterLogos(guiPath);

    // Num samples, threads / block and max block adjustables
    ui->numSampleBox->setMinimum(1024);
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

void MainApp::showDropdown1()
{
    ui->comboBox->showPopup();
}

void MainApp::showDropdown2()
{
    ui->comboBox_2->showPopup();
}

void MainApp::showDropdown3()
{
    ui->comboBox_3->showPopup();
}

void MainApp::showDropdown4()
{
    ui->comboBox_4->showPopup();
}

void MainApp::showDropdown5()
{
    ui->comboBox_5->showPopup();
}

void MainApp::showDropdown6()
{
    ui->comboBox_6->showPopup();
}

/**
  * This helper function retrieves the current device's properties
  * in "NVidia GPU Properties" section.
  */
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
 * This helper function sets the song properties in the appropriate
 * display box.
 */
void MainApp::setSongProperties()
{
    WavData *song = paramEQ->getSong();
    QString fileSize;
    // Get overall size of the song in bytes
    int overallSize = song->actualSize + 36;
    // 1 MB = (1024 * 1024) bytes
    fileSize = QString::number((double)overallSize / (1024 * 1024), 'f', 2);
    fileSize = "File Size: " + fileSize + " MB";

    QString numCh = QString::number(song->numChannels);
    numCh = "Number of Channels: " + numCh;

    QString samplingRate = QString::number(song->samplingRate);
    samplingRate = "Sampling Rate: " + samplingRate + " Hz";

    QString bitsPSample = QString::number(song->bitsPerSample);
    bitsPSample = "Bits Per Sample: " + bitsPSample;

    QString songProp = fileSize + "\n" + numCh + "\n" + samplingRate +
        "\n" + bitsPSample;
    ui->songBrowser->setText(songProp);
}

/**
 * This helper function calculates a time string from a time (which
 * represents a song's length) in seconds.
 */
QString MainApp::calculateTimeString(float timeFloat)
{
    int seconds, hours, minutes;
    
    // Convert to an int
    int time = (int) std::ceil(timeFloat);
    
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
void MainApp::setNewDuration(float newDuration)
{
    duration = newDuration;
    alreadyPlayed = 0.0;
    setTimeString();
    
    ui->progressBar->setRange(0.0, 
            std::ceil(duration * PROG_BAR_RES_PER_S));
    ui->progressBar->setValue(0.0);
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
    if (filename.isEmpty() || 
        !(filename.contains(".wav") || filename.contains(".WAV")) )
    {
        // Display an error message only if a file wasn't selected.
        if (!filename.isEmpty())
        {
            QMessageBox::information(this, tr("Invalid File Format"),
                "\"" + filename + "\" is not a valid WAV file!");
        }
        
        // Set filename to none
        filename = "";
    }
    else
    {
        ui->textEdit->setText(filename);
        currDataFile = filename;
        char* charDataPath  = currDataFile.toLocal8Bit().data();
        
        // Initialize properties on the Parametric EQ's side.
        paramEQ->setSong(charDataPath);
        paramEQ->setNumBufSamples(numSamples, filters);
        paramEQ->setThreadsPerBlock(threadNumPerBlock);
        paramEQ->setMaxBlocks(maxNumBlock);

        // Read the song's duration and update the time string.
        float newDuration = (paramEQ->getSong())->duration();
        setNewDuration(newDuration);

        // Set song properties.
        setSongProperties();
    }
}


/**
 * Set knob's LCD number value based on the direction that a knob turned.
 * Also, update the set of filters.
 */
void MainApp::setKnobValue(int knobNum, int val)
{
    // Frequency = FREQ_MULT * LOG_BASE^(knobValue / EXP_DIV)
    dialValue[knobNum] = std::round(QCustomDial::FREQ_MULT * 
            std::pow(QCustomDial::LOG_BASE, val / QCustomDial::EXP_DIV));
    
    // cout << "Set dialValue[" << knobNum << "] to " << dialValue[knobNum]
    //    << endl;
    
    // Find the corresponding filter number that's been affected.
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
    
    // Update gain array.
    gain[filterNum] = val;
   
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
 * This helper function is called when we want to stop processing. It does
 * everything except for actually call stopProcessingSound() on the
 * ParametricEQ (this must be done elsewhere).
 */
void MainApp::handleStopProcessing()
{
    // Clean up after the processing thread.
    if (processingThread != NULL)
    {
        processingThread->join();
        delete processingThread;
        processingThread = NULL;
    }

    // Stop the relevant timers.
    if (songUpdatesTimer != NULL)
    {
        songUpdatesTimer->stop();
        delete songUpdatesTimer;
        songUpdatesTimer = NULL;
    }

    // Clean up the GUI
    ui->processButton->setText("Process");
    ui->fileSelectButton->setEnabled(true);
    ui->threadsBlockBox->setEnabled(true);
    ui->numSampleBox->setEnabled(true);
    ui->blockNum->setEnabled(true);

    processing = false;
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

    cout << "Finished processing file: " << currDataFile.toLocal8Bit().data() 
         << endl;
    
    // Signal to songListener() that we're done.
    processing = false;
}


/**
 * This function is called by the "songUpdatesTimer" so we can update
 * things based on how much of the song has played. It is used to figure
 * out how much of the song we've played, and also to check if we've
 * stopped processing without the "Stop" button being processed.
 *
 */
void MainApp::songListener()
{
    // Check if we stopped processing.
    if (!processing)
    {
        // Clean up the timer and the GUI
        handleStopProcessing();
        return;
    }
    
    // Update the progress bar and the time string.
    
    // Time played in seconds (float).
    alreadyPlayed = paramEQ->getPlayedTime();
    setTimeString();
    
    // Account for the resolution of the progress bar.
    ui->progressBar->setValue(
            std::ceil(alreadyPlayed * PROG_BAR_RES_PER_S));
    ui->progressBar->repaint();
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
        paramEQ->stopProcessingSound();

        cout << "Stopped processing file: " << 
            currDataFile.toLocal8Bit().data() << endl;
        
        // Clean up the GUI, timers, processing thread, etc.
        handleStopProcessing();
    }
    else
    {
        // Start processing
        
        // Update GUI
        ui->processButton->setText("Stop");
        ui->fileSelectButton->setEnabled(false);
        ui->threadsBlockBox->setEnabled(false);
        ui->numSampleBox->setEnabled(false);
        ui->blockNum->setEnabled(false);

        cout << "Processing file: " << currDataFile.toLocal8Bit().data() 
             << endl;
        
        processing = true;

        // Use a separate thread to start processing.
        processingThread = new boost::thread(boost::bind(
                    &MainApp::initiateProcessing, this));
            
        // Timer to listen to song updates, and call songListener() every 
        // LISTENER_UPD_MS milliseconds.
        songUpdatesTimer = new QTimer(this);
        connect(songUpdatesTimer, SIGNAL(timeout()), this, 
                SLOT(songListener()));
        songUpdatesTimer->start(LISTENER_UPD_MS);
        
    }
    
}


void MainApp::on_numSampleBox_editingFinished()
{
    int newNumSamples = ui->numSampleBox->value();
    
    paramEQ->setNumBufSamples(newNumSamples, filters);
    numSamples = newNumSamples;

    // Display a status message in the bar and in the terminal
    QString msg = QString("The number of samples to use per buffer ") +
                  QString("per channel) has been set to ") + 
                  QString::number(newNumSamples);
   
#ifndef NDEBUG
    cout << msg.toUtf8().constData() << endl;
#endif

    ui->statusBar->showMessage(msg, 5000);
}


void MainApp::on_threadsBlockBox_editingFinished()
{
    int newThreadsBlock = ui->threadsBlockBox->value();
    
    paramEQ->setThreadsPerBlock(newThreadsBlock);
    threadNumPerBlock = newThreadsBlock;
    
    // Display a status message in the bar and in the terminal
    QString msg = QString("Threads per block has been set to ") +
                  QString::number(threadNumPerBlock);
    
#ifndef NDEBUG
    cout << msg.toUtf8().constData() << endl;
#endif

    ui->statusBar->showMessage(msg, 5000);
}


void MainApp::on_blockNum_editingFinished()
{
    int newBlockNum = ui->blockNum->value();
    
    paramEQ->setMaxBlocks(newBlockNum);
    maxNumBlock = newBlockNum;
    
    // Display a status message in the bar and in the terminal.
    QString msg = QString("Max number of blocks has been set to ") +
                  QString::number(maxNumBlock);
    
#ifndef NDEBUG
    cout << msg.toUtf8().constData() << endl;
#endif

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

    // Update the plot
    if (plotInitialized)
    {
        updatePlot();
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


/** Destructor **/
MainApp::~MainApp()
{
    freeFilterProperties(); 

    delete paramEQ;
    paramEQ = NULL;

    delete ui;
    ui = NULL;
}


