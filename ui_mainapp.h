/********************************************************************************
** Form generated from reading UI file 'mainapp.ui'
**
** Created by: Qt User Interface Compiler version 5.4.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINAPP_H
#define UI_MAINAPP_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDial>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainApp
{
public:
    QWidget *centralWidget;
    QPushButton *fileSelectButton;
    QTextEdit *textEdit;
    QPushButton *processButton;
    QWidget *horizontalLayoutWidget;
    QHBoxLayout *horizontalLayout;
    QSlider *verticalSlider;
    QSlider *verticalSlider_2;
    QSlider *verticalSlider_3;
    QSlider *verticalSlider_4;
    QSlider *verticalSlider_5;
    QSlider *verticalSlider_6;
    QProgressBar *progressBar;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QGraphicsView *graphicsView;
    QTextBrowser *textBrowser;
    QLabel *label_4;
    QLabel *label_5;
    QLabel *appLogo;
    QSpinBox *threadsBlockBox;
    QSpinBox *blockNum;
    QDial *dial_4;
    QDial *dial_5;
    QDial *dial_1;
    QDial *dial_3;
    QDial *dial_2;
    QDial *dial_6;
    QDial *dial_7;
    QDial *dial_12;
    QDial *dial_11;
    QDial *dial_9;
    QDial *dial_8;
    QDial *dial_10;
    QLabel *label_6;
    QMenuBar *menuBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainApp)
    {
        if (MainApp->objectName().isEmpty())
            MainApp->setObjectName(QStringLiteral("MainApp"));
        MainApp->resize(1017, 559);
        centralWidget = new QWidget(MainApp);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        fileSelectButton = new QPushButton(centralWidget);
        fileSelectButton->setObjectName(QStringLiteral("fileSelectButton"));
        fileSelectButton->setGeometry(QRect(660, 10, 101, 31));
        textEdit = new QTextEdit(centralWidget);
        textEdit->setObjectName(QStringLiteral("textEdit"));
        textEdit->setGeometry(QRect(10, 10, 631, 41));
        processButton = new QPushButton(centralWidget);
        processButton->setObjectName(QStringLiteral("processButton"));
        processButton->setGeometry(QRect(770, 10, 231, 31));
        horizontalLayoutWidget = new QWidget(centralWidget);
        horizontalLayoutWidget->setObjectName(QStringLiteral("horizontalLayoutWidget"));
        horizontalLayoutWidget->setGeometry(QRect(680, 130, 311, 251));
        horizontalLayout = new QHBoxLayout(horizontalLayoutWidget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        verticalSlider = new QSlider(horizontalLayoutWidget);
        verticalSlider->setObjectName(QStringLiteral("verticalSlider"));
        verticalSlider->setOrientation(Qt::Vertical);

        horizontalLayout->addWidget(verticalSlider);

        verticalSlider_2 = new QSlider(horizontalLayoutWidget);
        verticalSlider_2->setObjectName(QStringLiteral("verticalSlider_2"));
        verticalSlider_2->setOrientation(Qt::Vertical);

        horizontalLayout->addWidget(verticalSlider_2);

        verticalSlider_3 = new QSlider(horizontalLayoutWidget);
        verticalSlider_3->setObjectName(QStringLiteral("verticalSlider_3"));
        verticalSlider_3->setOrientation(Qt::Vertical);

        horizontalLayout->addWidget(verticalSlider_3);

        verticalSlider_4 = new QSlider(horizontalLayoutWidget);
        verticalSlider_4->setObjectName(QStringLiteral("verticalSlider_4"));
        verticalSlider_4->setOrientation(Qt::Vertical);

        horizontalLayout->addWidget(verticalSlider_4);

        verticalSlider_5 = new QSlider(horizontalLayoutWidget);
        verticalSlider_5->setObjectName(QStringLiteral("verticalSlider_5"));
        verticalSlider_5->setOrientation(Qt::Vertical);

        horizontalLayout->addWidget(verticalSlider_5);

        verticalSlider_6 = new QSlider(horizontalLayoutWidget);
        verticalSlider_6->setObjectName(QStringLiteral("verticalSlider_6"));
        verticalSlider_6->setOrientation(Qt::Vertical);

        horizontalLayout->addWidget(verticalSlider_6);

        progressBar = new QProgressBar(centralWidget);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setGeometry(QRect(10, 70, 631, 21));
        progressBar->setValue(24);
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(356, 90, 281, 20));
        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(640, 410, 41, 20));
        label_3 = new QLabel(centralWidget);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(650, 450, 31, 20));
        graphicsView = new QGraphicsView(centralWidget);
        graphicsView->setObjectName(QStringLiteral("graphicsView"));
        graphicsView->setGeometry(QRect(20, 130, 621, 251));
        textBrowser = new QTextBrowser(centralWidget);
        textBrowser->setObjectName(QStringLiteral("textBrowser"));
        textBrowser->setGeometry(QRect(190, 410, 161, 91));
        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(370, 480, 111, 21));
        label_5 = new QLabel(centralWidget);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(500, 480, 121, 21));
        appLogo = new QLabel(centralWidget);
        appLogo->setObjectName(QStringLiteral("appLogo"));
        appLogo->setGeometry(QRect(-20, 380, 171, 151));
        threadsBlockBox = new QSpinBox(centralWidget);
        threadsBlockBox->setObjectName(QStringLiteral("threadsBlockBox"));
        threadsBlockBox->setGeometry(QRect(370, 450, 111, 27));
        blockNum = new QSpinBox(centralWidget);
        blockNum->setObjectName(QStringLiteral("blockNum"));
        blockNum->setGeometry(QRect(500, 450, 111, 27));
        dial_4 = new QDial(centralWidget);
        dial_4->setObjectName(QStringLiteral("dial_4"));
        dial_4->setEnabled(true);
        dial_4->setGeometry(QRect(840, 400, 45, 29));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(dial_4->sizePolicy().hasHeightForWidth());
        dial_4->setSizePolicy(sizePolicy);
        dial_5 = new QDial(centralWidget);
        dial_5->setObjectName(QStringLiteral("dial_5"));
        dial_5->setEnabled(true);
        dial_5->setGeometry(QRect(890, 400, 45, 29));
        sizePolicy.setHeightForWidth(dial_5->sizePolicy().hasHeightForWidth());
        dial_5->setSizePolicy(sizePolicy);
        dial_1 = new QDial(centralWidget);
        dial_1->setObjectName(QStringLiteral("dial_1"));
        dial_1->setEnabled(true);
        dial_1->setGeometry(QRect(690, 400, 45, 29));
        sizePolicy.setHeightForWidth(dial_1->sizePolicy().hasHeightForWidth());
        dial_1->setSizePolicy(sizePolicy);
        dial_3 = new QDial(centralWidget);
        dial_3->setObjectName(QStringLiteral("dial_3"));
        dial_3->setEnabled(true);
        dial_3->setGeometry(QRect(790, 400, 45, 29));
        sizePolicy.setHeightForWidth(dial_3->sizePolicy().hasHeightForWidth());
        dial_3->setSizePolicy(sizePolicy);
        dial_2 = new QDial(centralWidget);
        dial_2->setObjectName(QStringLiteral("dial_2"));
        dial_2->setEnabled(true);
        dial_2->setGeometry(QRect(740, 400, 45, 29));
        sizePolicy.setHeightForWidth(dial_2->sizePolicy().hasHeightForWidth());
        dial_2->setSizePolicy(sizePolicy);
        dial_6 = new QDial(centralWidget);
        dial_6->setObjectName(QStringLiteral("dial_6"));
        dial_6->setEnabled(true);
        dial_6->setGeometry(QRect(940, 400, 45, 29));
        sizePolicy.setHeightForWidth(dial_6->sizePolicy().hasHeightForWidth());
        dial_6->setSizePolicy(sizePolicy);
        dial_7 = new QDial(centralWidget);
        dial_7->setObjectName(QStringLiteral("dial_7"));
        dial_7->setEnabled(true);
        dial_7->setGeometry(QRect(690, 440, 45, 29));
        sizePolicy.setHeightForWidth(dial_7->sizePolicy().hasHeightForWidth());
        dial_7->setSizePolicy(sizePolicy);
        dial_12 = new QDial(centralWidget);
        dial_12->setObjectName(QStringLiteral("dial_12"));
        dial_12->setEnabled(true);
        dial_12->setGeometry(QRect(940, 440, 45, 29));
        sizePolicy.setHeightForWidth(dial_12->sizePolicy().hasHeightForWidth());
        dial_12->setSizePolicy(sizePolicy);
        dial_11 = new QDial(centralWidget);
        dial_11->setObjectName(QStringLiteral("dial_11"));
        dial_11->setEnabled(true);
        dial_11->setGeometry(QRect(890, 440, 45, 29));
        sizePolicy.setHeightForWidth(dial_11->sizePolicy().hasHeightForWidth());
        dial_11->setSizePolicy(sizePolicy);
        dial_9 = new QDial(centralWidget);
        dial_9->setObjectName(QStringLiteral("dial_9"));
        dial_9->setEnabled(true);
        dial_9->setGeometry(QRect(790, 440, 45, 29));
        sizePolicy.setHeightForWidth(dial_9->sizePolicy().hasHeightForWidth());
        dial_9->setSizePolicy(sizePolicy);
        dial_8 = new QDial(centralWidget);
        dial_8->setObjectName(QStringLiteral("dial_8"));
        dial_8->setEnabled(true);
        dial_8->setGeometry(QRect(740, 440, 45, 29));
        sizePolicy.setHeightForWidth(dial_8->sizePolicy().hasHeightForWidth());
        dial_8->setSizePolicy(sizePolicy);
        dial_10 = new QDial(centralWidget);
        dial_10->setObjectName(QStringLiteral("dial_10"));
        dial_10->setEnabled(true);
        dial_10->setGeometry(QRect(840, 440, 45, 29));
        sizePolicy.setHeightForWidth(dial_10->sizePolicy().hasHeightForWidth());
        dial_10->setSizePolicy(sizePolicy);
        label_6 = new QLabel(centralWidget);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(10, 510, 261, 16));
        QFont font;
        font.setPointSize(9);
        font.setBold(true);
        font.setWeight(75);
        label_6->setFont(font);
        MainApp->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainApp);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1017, 25));
        MainApp->setMenuBar(menuBar);
        statusBar = new QStatusBar(MainApp);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainApp->setStatusBar(statusBar);

        retranslateUi(MainApp);

        QMetaObject::connectSlotsByName(MainApp);
    } // setupUi

    void retranslateUi(QMainWindow *MainApp)
    {
        MainApp->setWindowTitle(QApplication::translate("MainApp", "MainApp", 0));
        fileSelectButton->setText(QApplication::translate("MainApp", "Browse", 0));
        processButton->setText(QApplication::translate("MainApp", "Process", 0));
        label->setText(QApplication::translate("MainApp", "0:00 / 3:00", 0));
        label_2->setText(QApplication::translate("MainApp", "FREQ", 0));
        label_3->setText(QApplication::translate("MainApp", "BW", 0));
        textBrowser->setHtml(QApplication::translate("MainApp", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Nvidia GTX 770</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Architecture Kepler</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">CC: 3.0</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">CUDA Version: 7.0</p></body></html>", 0));
        label_4->setText(QApplication::translate("MainApp", "Threads / Block", 0));
        label_5->setText(QApplication::translate("MainApp", "Max. # of Blocks", 0));
        appLogo->setText(QString());
        label_6->setText(QApplication::translate("MainApp", "By Laksh Bhasin and Sharon Yang", 0));
    } // retranslateUi

};

namespace Ui {
    class MainApp: public Ui_MainApp {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINAPP_H
