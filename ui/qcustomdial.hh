/**
 * Custom dial code for a dial that takes on logarithmically spaced values.
 * Adapted from the example code in:
 *      http://thecodeinn.blogspot.com/2015/03/customizing-qdials-in-qt-part-2.html
 *
 */

#ifndef QCUSTOMDIAL_HH
#define QCUSTOMDIAL_HH
 
/* Standard includes */
#include <cmath>
#include <iostream>

/* Qt includes */
#include <QDial>
#include <QString>
#include <QSharedPointer>
 
class QColor;
class QRectF;
class QPen;

using std::cout;
using std::cerr;
using std::endl;
 
class QCustomDial : public QDial
{
    Q_OBJECT
     
    Q_PROPERTY(QString arcColor READ getArcColor WRITE setArcColor)
     
    Q_PROPERTY(double arcWidth READ getArcWidth WRITE setArcWidth)
     
public:

    // The multiplier to use for frequencies. Note that frequencies from
    // the knob are given by FREQ_MULT * LOG_BASE^(knobValue / EXP_DIV),
    // where knobValue is the knob's value (which ranges from KNOB_MIN to
    // KNOB_MAX).
    static constexpr float FREQ_MULT = 16.0;   

    // The base to use for the logarithmically-spaced knobs.
    static constexpr float LOG_BASE = 2.0;

    // The number to divide by in the exponent, when computing
    // logarithmically-spaced frequencies.
    static constexpr float EXP_DIV = 50.0;
    
    explicit QCustomDial(QWidget* parent = nullptr);
     
    void setArcColor(const QString& color);
    QString getArcColor() const;
     
    void setStartAngle(double angle);
    double getStartAngle() const;
     
    void setMaximumAngle(double angle);
    double getMaximumAngle() const;
     
    void setArcWidth(double px);
    double getArcWidth() const;

    void setFrequency(int freq);
    int getFrequency();
     
    ~QCustomDial();
     
private slots:
     
    void updateValue();
     
private:

    // The minimum and maximum knob values to allow.
    static constexpr int KNOB_MIN = 0;
    static constexpr int KNOB_MAX = 500;

    // The actual frequency, given by rounding 
    // FREQ_MULT * LOG_BASE^(knobValue / EXP_DIV)
    int frequency;
    
    double maximumAngleSpan_;
    double startAngle_;
    double arcWidth_;
    double angleSpan_;
     
    QString valueString_;
    
    QSharedPointer<QRectF> arcRect_;
    QSharedPointer<QRectF> valueRect_;
    QSharedPointer<QColor> arcColor_;
    QSharedPointer<QPen> arcPen_;

    virtual void paintEvent(QPaintEvent*) override;
    virtual void resizeEvent(QResizeEvent* event) override;
};
 
#endif // QCUSTOMDIAL_HH
