#include "mainapp.hh"
#include "ui_mainapp.hh"
#include <QFileDialog>
#include <QMessageBox>

MainApp::MainApp(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainApp)
{
    ui->setupUi(this);
    initWindow();
}

void MainApp::initWindow()
{
    // Set window name
    QMainWindow::setWindowTitle("GPU EQ");
    QMainWindow::setFixedWidth(1020);
    QMainWindow::setFixedHeight(560);

    // Find gui path for logo
    QString guiPath = QDir::currentPath().mid(
        0, QDir::currentPath().indexOf("gpu_parametric_eq") + 17) + "/gui/";

    // Get logo
    QPixmap logo(guiPath + "gpu_logo.gif");
    ui->label_logo->setPixmap(logo);

    // Threads / block and max block adjustables
    ui->spinBox->setMaximum(8192);
    ui->spinBox_2->setMaximum(1000);
    ui->spinBox->setValue(512);
    ui->spinBox_2->setValue(200);


}

MainApp::~MainApp()
{
    delete ui;
}

void MainApp::on_pushButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
        this, tr("Open WAV file for test"), tr("Audio files (*.wav)"));

    // Check for .wav or .WAV extension
    if (! (filename.contains(".wav") || filename.contains(".WAV")) )
    {
        QMessageBox::information(this, tr("Error On selection"),
            "File name \"" + filename + "\" is not a valid WAV file!");
        // Set filename to none
        filename = "";
    }
    else
    {
        ui->textEdit->setText(filename);
    }

}

