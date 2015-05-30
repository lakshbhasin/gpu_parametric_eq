/**
 * Code modified from:
 *      http://thecodeinn.blogspot.com/2015/03/customizing-qdials-in-qt-part-2.html
 *
 */

#include "qcustomdial.hh"
 
#include <QPainter>
#include <QColor>
#include <QLabel>
#include <QRectF>
#include <QPen>
#include <QResizeEvent>
 
QCustomDial::QCustomDial(QWidget* parent) : 
    QDial(parent), frequency(0), arcRect_(new QRectF), 
    valueRect_(new QRectF), arcColor_(new QColor), arcPen_(new QPen) 
{
    // This sets the range of the knob, **not** the frequency range. The
    // frequency range is given by [FREQ_MULT * LOG_BASE^(KNOB_MIN),
    // FREQ_MULT * LOG_BASE^(KNOB_MAX)].
    QDial::setRange(KNOB_MIN, KNOB_MAX);
    
    QDial::setCursor(Qt::PointingHandCursor);
    
    // Have our updateValue() function get notified when values change.
    connect(this, &QDial::valueChanged,
            this, &QCustomDial::updateValue);
    
    // Arbitary; depends on how big the "value" is, etc.
    setMinimumSize(30, 30);

    // Measured relative to positive x-axis. Positive => counterclockwise.
    setStartAngle(225);
    
    // Measured relative to start angle. Negative => clockwise.
    setMaximumAngle(-270);

    // Disallow wrapping.
    setWrapping(false);     

    updateValue();
}
 
QCustomDial::~QCustomDial() = default;
 
void QCustomDial::paintEvent(QPaintEvent*)
{
    QPainter painter(this);
     
    // So that we can use the background color. Otherwise the background
    // is transparent
    painter.setBackgroundMode(Qt::OpaqueMode);
     
    // Smooth out the circle
    painter.setRenderHint(QPainter::Antialiasing);
     
    // Use background color
    painter.setBrush(painter.background());
     
    // Get current pen before resetting so we have access to the color()
    // method which returns the color from the stylesheet.
    QPen textPen = painter.pen();
     
    // No border
    painter.setPen(QPen(Qt::NoPen));
     
    // Draw background circle
    painter.drawEllipse(QDial::rect());
    
    painter.setPen(textPen);
    
    painter.drawText(*valueRect_, Qt::AlignHCenter | Qt::AlignVCenter, 
                     valueString_);
     
    painter.setPen(*arcPen_);
     
    painter.drawArc(*arcRect_, startAngle_, angleSpan_);
     
}
 
void QCustomDial::resizeEvent(QResizeEvent* event)
{
    QDial::setMinimumSize(event->size());
     
    double width = QDial::width() - (2 * arcWidth_);
     
    double height = width / 2;
     
    *valueRect_ = QRectF(0, 0, QDial::width(), QDial::height());
     
    *arcRect_ = QRectF(arcWidth_ / 2,
                       arcWidth_ / 2,
                       QDial::width() - arcWidth_,
                       QDial::height() - arcWidth_);
}


/**
 * This function is exposed to the user, not to Qt. It lets us set the knob
 * value for a given frequency. Usually, this is only used for
 * initialization.
 *
 * Frequencies are given by FREQ_MULT * LOG_BASE^(knobValue / EXP_DIV), so
 * we can get the knob value via:
 *      
 *      knobValue = log_{LOG_BASE}(freq / FREQ_MULT) * EXP_DIV
 *
 */
void QCustomDial::setFrequency(int freq)
{
    // Calculate the new knob value.
    int newKnobValue = std::round(EXP_DIV * std::log(freq / FREQ_MULT) /
        std::log(LOG_BASE));

    // Set the knob visually and calculate the new, rounded frequency.
    setValue(newKnobValue);
}


int QCustomDial::getFrequency()
{
    return frequency;
}


void QCustomDial::updateValue()
{
    // Override the settings imposed by QtCreator if necessary...
    if (QDial::maximum() != KNOB_MAX || QDial::minimum() != KNOB_MIN)
    {
        QDial::setRange(KNOB_MIN, KNOB_MAX);
    }

    // Get the knob's value (not equal to the frequency)
    float knobValue = QDial::value();
    
    // Get ratio between current value and maximum to calculate angle
    float ratio = knobValue / QDial::maximum();
    angleSpan_ = maximumAngleSpan_ * ratio;
    
    // Calculate the frequency.
    frequency = std::round(FREQ_MULT * 
            std::pow(LOG_BASE, knobValue / EXP_DIV));
    
    // Update the "value" string. The format of this depends on the actual
    // frequency.
    
    if (frequency >= 0 && frequency <= 999)
    {
        valueString_ = QString::number(frequency);
    }
    else if (frequency >= 1000 && frequency <= 9999)
    {
        // Example: 1024 will be displayed as "1.02k" (it will be stored
        // internally as 1024, though).
        int thousandsPlace = frequency / 1000;
        int leftover = frequency % 1000;

        // Round "leftover" to the nearest tens place.
        int remainder = leftover % 10;

        if (remainder >= 5)
        {
            // Round up
            leftover = leftover - remainder + 10;
        }
        else
        {
            // Round down
            leftover = leftover - remainder;
        }

        // We might need to round to the next thousands place.
        if (leftover == 1000)
        {
            leftover = 0;
            thousandsPlace ++;
        }

        // Get a string representing the leftover tens place. For example,
        // for 1024, this will be "02k".
        QString tensPlaceString = "";

        if (leftover < 10)
        {
            // Ignore leftover digits
            tensPlaceString = QString("00k");
        }
        else if (leftover >= 10 && leftover < 100)
        {
            tensPlaceString = QString("0") + QString::number(leftover / 10)
                + QString("k");
        }
        else
        {
            tensPlaceString = QString::number(leftover / 10) 
                + QString("k");
        }

        // Assemble the final string.
        valueString_ = QString::number(thousandsPlace) + QString(".") +
            tensPlaceString;
    }
    else
    {
        // Example: 16384 will be displayed as "16.4k" (it will still be
        // stored internally as 16384, though).
        int thousandsPlace = frequency / 1000;
        int leftover = frequency % 1000;

        // Round "leftover" to the nearest hundreds place.
        int remainder = leftover % 100;

        if (remainder >= 50)
        {
            // Round up
            leftover = leftover - remainder + 100;
        }
        else
        {
            // Round down
            leftover = leftover - remainder;
        }

        // We might need to round to the next thousands place.
        if (leftover == 1000)
        {
            leftover = 0;
            thousandsPlace ++;
        }

        // Get a string representing the leftover hundreds place. For 
        // example, for 16384, this will be "4k".
        QString hundredsPlaceString = QString::number(leftover / 100)
            + QString("k");

        // Assemble the final string.
        valueString_ = QString::number(thousandsPlace) + QString(".") +
            hundredsPlaceString;
    }
    
}
 
void QCustomDial::setArcWidth(double px)
{
    arcWidth_ = px;
     
    *arcRect_ = QRectF(arcWidth_ / 2,
                       arcWidth_ / 2,
                       QDial::width() - arcWidth_,
                       QDial::height() - arcWidth_);
     
    arcPen_->setWidth(arcWidth_);
}
 
double QCustomDial::getArcWidth() const
{
    return arcWidth_;
}
 
void QCustomDial::setMaximumAngle(double angle)
{
    maximumAngleSpan_ = angle * 16;
}
 
double QCustomDial::getMaximumAngle() const
{
    return maximumAngleSpan_ / 16;
}
 
void QCustomDial::setStartAngle(double angle)
{
    startAngle_ = angle * 16;
}
 
double QCustomDial::getStartAngle() const
{
    return startAngle_ / 16;
}
 
void QCustomDial::setArcColor(const QString& color)
{
    arcColor_->setNamedColor(color);
     
    arcPen_->setColor(*arcColor_);
}
 
QString QCustomDial::getArcColor() const
{
    return arcColor_->name();
}
