#include "mainapp.hh"
#include "ui_mainapp.h"
#include <QFileDialog>
#include <QMessageBox>

#include "parametric_eq.hh"

MainApp::MainApp(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainApp)
{
    ui->setupUi(this);
    initWindow();
}

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
    ui->threadsBlockBox->setMaximum(8192);
    ui->threadsBlockBox->setValue(512);
    ui->blockNum->setMaximum(1000);
    ui->blockNum->setValue(200);

}

MainApp::~MainApp()
{
    delete ui;
}

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

void MainApp::setTimeString()
{
    QString played = calculateTimeString(alreadyPlayed);
    QString fullLength = calculateTimeString(duration);
    QString playLabel = played + " / " + fullLength;
    ui->label->setText(playLabel);
}

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
        deploy(charDataPath , 4096, 512, 200);
        duration = song->duration();
        alreadyPlayed = 0;
        setTimeString();
    }
}

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

    ui->processButton->setText("Stop");
    cout << "Processing file: " << currDataFile.toLocal8Bit().data() << endl;
    processSound();
}
