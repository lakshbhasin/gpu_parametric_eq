#ifndef MAINAPP_HH
#define MAINAPP_HH

#include <QMainWindow>
#include <QFileDialog>

namespace Ui {
class MainApp;
}

class MainApp : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainApp(QWidget *parent = 0);
    ~MainApp();

private slots:
    void on_fileSelectButton_clicked();

    void on_processButton_clicked();

private:
    QString currDataFile;
    int alreadyPlayed;
    int duration;
    Ui::MainApp *ui;
    void initWindow();
    QString calculateTimeString(int time);
    void setTimeString();
};

#endif // MAINAPP_HH
