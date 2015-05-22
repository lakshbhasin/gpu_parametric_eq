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
    void on_pushButton_clicked();

private:
    Ui::MainApp *ui;
    void initWindow();
};

#endif // MAINAPP_HH
