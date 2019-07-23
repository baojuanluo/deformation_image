#include "opencv2/opencv.hpp"

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <io.h>

using namespace cv;
using namespace std;

typedef struct StDeformation
{
  Point srcpt;
  Point dstpt;
  Vec3b vb;
};

RNG rng((unsigned)time(NULL));


void get_files(std::string path, std::vector<std::string> vecsuffix, std::vector<std::string>& files, std::string prefix)
{
  path = path + "/*.*";
  _finddata_t file;
  long long lf = _findfirst(path.c_str(), &file);
  //输入文件夹路径  
  if (lf == -1)
  {
    std::cout << path << " not found!!!" << std::endl;
  }
  else
  {
    while (_findnext(lf, &file) == 0)
    {
      if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
      {
        continue;
      }
      std::string strfilename = file.name;

      if ((prefix != "" && 0 == strfilename.find(prefix)) || prefix == "")
      {
        int npos = strfilename.find_last_of(".");
        std::string strsuffix = strfilename.substr(npos + 1);
        transform(strsuffix.begin(), strsuffix.end(), strsuffix.begin(), ::tolower);
        for (int i = 0; i < vecsuffix.size(); i++)
        {
          if (0 == strsuffix.compare(vecsuffix[i]))
          {
            files.push_back(file.name);
            break;
          }
        }
      }

    }
  }
  _findclose(lf);
}

float getDist_P2L(CvPoint pointP, CvPoint pointA, CvPoint pointB)
{ 
  //求直线方程 
  int A = 0, B = 0, C = 0; 
  A = pointA.y - pointB.y; 
  B = pointB.x - pointA.x; 
  C = pointA.x * pointB.y - pointA.y * pointB.x; 
  //代入点到直线距离公式 
  float distance = 0; 
  distance = ((float)abs(A * pointP.x + B * pointP.y + C)) / ((float)sqrtf(A * A + B * B));
  return distance; 
}


void interpolat_img(Mat& bwimg, Mat& image, int interpolationsize)
{
  Mat bwimgd = bwimg.clone();
  imwrite("bwimg.jpg", bwimg);
  dilate(bwimgd, bwimgd, getStructuringElement(MORPH_CROSS, Size(10, 10)));
  erode(bwimgd, bwimgd, getStructuringElement(MORPH_CROSS, Size(10, 10)));
  imwrite("bwimgd.jpg", bwimgd);

  cout << "hello" << endl;

  for (int j = 0; j < bwimg.rows; j++)
  {
    for (int i = 0; i < bwimg.cols; i++)
    {
      if (bwimg.at<uchar>(j, i) == 0 && bwimgd.at<uchar>(j, i) == 255)
      {
        int cnt = 0;
        Vec3i valsum(0, 0, 0);
        vector<Vec3b> vvb;
        for (int m = -1 * interpolationsize; m <= interpolationsize; m++)
        {
          for (int n = -1 * interpolationsize; n <= interpolationsize; n++)
          {
            int rowindex = j + m;
            int colindex = i + n;
            if (rowindex < 0)
            {
              rowindex = 0;
            }
            if (colindex < 0)
            {
              colindex = 0;
            }
            if (rowindex > bwimg.rows - 1)
            {
              rowindex = bwimg.rows - 1;
            }
            if (colindex > bwimg.cols - 1)
            {
              colindex = bwimg.cols - 1;
            }
            if (bwimg.at<uchar>(rowindex, colindex) != 0)
            {
              cnt++;
              Vec3b vb = image.at<Vec3b>(rowindex, colindex);
              valsum[0] = valsum[0] + vb[0];
              valsum[1] = valsum[1] + vb[1];
              valsum[2] = valsum[2] + vb[2];
              vvb.push_back(vb);
            }
          }
        }
        if (cnt > 0)
        {
          image.at<Vec3b>(j, i) = Vec3b(valsum[0] / cnt, valsum[1] / cnt, valsum[2] / cnt);
          //image.at<Vec3b>(j, i) = vvb[vvb.size() * 0.5];
        }
      }
    }
  }
}


void generate_flod(Mat& image, Mat& dmat, Point& v, float a, int idx, string savepath, string timename)
{
  //3.2根据公式生成w
  Mat foldw(image.size(), CV_32FC1, Scalar::all(0));
  for (int j = 0; j < dmat.rows; j++)
  {
    for (int i = 0; i < dmat.cols; i++)
    {
      foldw.at<float>(j, i) = float(a) / float(dmat.at<float>(j, i) + a);
    }
  }

  int flodcolmax = 0;
  int flodrowmax = 0;
  int flodcolmin = INT_MAX;
  int flodrowmin = INT_MAX;
  vector<StDeformation> vecfolddef;
  for (int j = 0; j < image.rows; j++)
  {
    for (int i = 0; i < image.cols; i++)
    {
      Point pi = Point(i, j) + foldw.at<float>(j, i) * v;
      StDeformation st;
      st.dstpt = pi;
      st.srcpt = Point(i, j);
      st.vb = image.at<Vec3b>(j, i);
      vecfolddef.push_back(st);
      if (pi.x > flodcolmax)
      {
        flodcolmax = pi.x;
      }
      if (pi.y > flodrowmax)
      {
        flodrowmax = pi.y;
      }
      if (pi.x < flodcolmin)
      {
        flodcolmin = pi.x;
      }
      if (pi.y < flodrowmin)
      {
        flodrowmin = pi.y;
      }
    }
  }

  if (flodcolmin == INT_MAX)
  {
    flodcolmin = 0;
  }
  if (flodrowmin == INT_MAX)
  {
    flodrowmin = 0;
  }
  Mat flodmat(Size(flodcolmax - flodcolmin + 1, flodrowmax - flodrowmin + 1), image.type(), Scalar::all(0));
  Mat flodbw(Size(flodcolmax - flodcolmin + 1, flodrowmax - flodrowmin + 1), CV_8UC1, Scalar::all(0));
  Mat flodx(Size(flodcolmax - flodcolmin + 1, flodrowmax - flodrowmin + 1), CV_8UC1, Scalar::all(0));
  Mat flody(Size(flodcolmax - flodcolmin + 1, flodrowmax - flodrowmin + 1), CV_8UC1, Scalar::all(0));
  auto itmap = vecfolddef.begin();
  while (itmap != vecfolddef.end())
  {
    //if (itmap->dstpt.x > 0 && itmap->dstpt.y > 0)
    {
      flodmat.at<Vec3b>(itmap->dstpt.y - flodrowmin, itmap->dstpt.x - flodcolmin) = itmap->vb;
      flodbw.at<uchar>(itmap->dstpt.y - flodrowmin, itmap->dstpt.x - flodcolmin) = 255;
      flodx.at<uchar>(itmap->dstpt.y - flodrowmin, itmap->dstpt.x - flodcolmin) = itmap->srcpt.x - itmap->dstpt.x + 128;
      flody.at<uchar>(itmap->dstpt.y - flodrowmin, itmap->dstpt.x - flodcolmin) = itmap->srcpt.y - itmap->dstpt.y + 128;
    }
    ++itmap;
  }

  FileStorage fs("flodx.yml", FileStorage::WRITE);
  fs << "trans" << flodx;
  fs.release();

  FileStorage fs1("flody.yml", FileStorage::WRITE);
  fs1 << "trans" << flody;
  fs1.release();

  /*normalize(flodx, flodx, 0, 255, CV_MINMAX, -1, flodbw);
  normalize(flody, flody, 0, 255, CV_MINMAX, -1, flodbw);*/
  
  int interpolationsize = 3;
  interpolat_img(flodbw, flodmat, interpolationsize);

  cv::imwrite(savepath + "dtc" + timename + "idx" + to_string(idx) + ".jpg", flodmat);
  cv::imwrite(savepath + "dtx" + timename + "idx" + to_string(idx) + ".jpg", flodx);
  cv::imwrite(savepath + "dty" + timename + "idx" + to_string(idx) + ".jpg", flody);
}


void generate_curves(Mat& image, Mat& dmat, Point& v, float a, int idx, string savepath, string timename)
{
  //3.2根据公式生成w
  Mat curvesw(image.size(), CV_32FC1, Scalar::all(0));
  for (int j = 0; j < dmat.rows; j++)
  {
    for (int i = 0; i < dmat.cols; i++)
    {
      curvesw.at<float>(j, i) = 1.0 - (float)pow(dmat.at<float>(j, i), a);
    }
  }

  vector<StDeformation> veccurvesdef;
  int curvescolmax = 0;
  int curvesrowmax = 0;
  int curvescolmin = INT_MAX;
  int curvesrowmin = INT_MAX;
  for (int j = 0; j < image.rows; j++)
  {
    for (int i = 0; i < image.cols; i++)
    {
      Point pi = Point(i, j) + curvesw.at<float>(j, i) * v;
      StDeformation st;
      st.srcpt = Point(i, j);
      st.dstpt = pi;
      st.vb = image.at<Vec3b>(j, i);
      veccurvesdef.push_back(st);
      if (pi.x > curvescolmax)
      {
        curvescolmax = pi.x;
      }
      if (pi.y > curvesrowmax)
      {
        curvesrowmax = pi.y;
      }
      if (pi.x < curvescolmin)
      {
        curvescolmin = pi.x;
      }
      if (pi.y < curvesrowmin)
      {
        curvesrowmin = pi.y;
      }
    }
  }

  if (curvescolmin == INT_MAX)
  {
    curvescolmin = 0;
  }
  if (curvesrowmin == INT_MAX)
  {
    curvesrowmin = 0;
  }

  Mat curvesmat(Size(curvescolmax - curvescolmin + 1, curvesrowmax - curvesrowmin + 1), image.type(), Scalar::all(0));
  Mat curvesbw(Size(curvescolmax - curvescolmin + 1, curvesrowmax - curvesrowmin + 1), CV_8UC1, Scalar::all(0));
  Mat curvesx(Size(curvescolmax - curvescolmin + 1, curvesrowmax - curvesrowmin + 1), CV_8UC1, Scalar::all(0));
  Mat curvesy(Size(curvescolmax - curvescolmin + 1, curvesrowmax - curvesrowmin + 1), CV_8UC1, Scalar::all(0));
  for (int i = 0; i < veccurvesdef.size(); i++)
  {
    //if (veccurvesdef[i].dstpt.y > 0 && veccurvesdef[i].dstpt.x > 0)
    {
      curvesmat.at<Vec3b>(veccurvesdef[i].dstpt.y - curvesrowmin, veccurvesdef[i].dstpt.x - curvescolmin) = veccurvesdef[i].vb;
      curvesbw.at<uchar>(veccurvesdef[i].dstpt.y - curvesrowmin, veccurvesdef[i].dstpt.x - curvescolmin) = 255;
      curvesx.at<uchar>(veccurvesdef[i].dstpt.y - curvesrowmin, veccurvesdef[i].dstpt.x - curvescolmin) = veccurvesdef[i].srcpt.x - veccurvesdef[i].dstpt.x + 128;
      curvesy.at<uchar>(veccurvesdef[i].dstpt.y - curvesrowmin, veccurvesdef[i].dstpt.x - curvescolmin) = veccurvesdef[i].srcpt.y - veccurvesdef[i].dstpt.y + 128;
    }
  }


  FileStorage fs("curvesx.yml", FileStorage::WRITE);
  fs << "trans" << curvesx;
  fs.release();

  FileStorage fs1("curvesy.yml", FileStorage::WRITE);
  fs1 << "trans" << curvesy;
  fs1.release();

  /*normalize(curvesx, curvesx, 0, 255, CV_MINMAX, -1, curvesbw);
  normalize(curvesy, curvesy, 0, 255, CV_MINMAX, -1, curvesbw);
  imwrite("curvesx.jpg", curvesx);
  imwrite("curvesy.jpg", curvesy);*/

  int interpolationsize = 3;
  interpolat_img(curvesbw, curvesmat, interpolationsize);

  cv::imwrite(savepath + "dtc" + timename + "idx" + to_string(idx) + ".jpg", curvesmat);
  cv::imwrite(savepath + "dtx" + timename + "idx" + to_string(idx) + ".jpg", curvesx);
  cv::imwrite(savepath + "dty" + timename + "idx" + to_string(idx) + ".jpg", curvesy);
}


void execute_deformation(Mat& image, string timename, string savepath)
{
  if (image.empty())
  {
    return;
  }


  if (image.channels() == 1)
  {
    cvtColor(image, image, CV_GRAY2BGR);
  }

  cout << "chaneels:" << image.channels() << endl;
  cout << "type:" << image.type() << endl;

  int idx = 0;
  for (int i = 0; i < 1; i++)
  {
    //1.生成随机点p
    Point p;
    p.x = abs(image.cols * rng.uniform(-1.0, 1.0));
    p.y = abs(image.rows * rng.uniform(-1.0, 1.0));
    cout << "p.x:" << p.x << endl;
    cout << "p.y" << p.y << endl;

    //2.随机生成变形方向和强度
    Point v;
    /*double xderection = rng.uniform(-1.0, 1.0);
    double yderection = rng.uniform(-1.0, 1.0);
    int dist = 128;
    if (xderection > 0)
    {
      if (image.cols - p.x < 128)
      {
        dist = image.cols - p.x;
      }
    }
    else
    {
      if (p.x < 128)
      {
        dist = p.x;
      }
    }
    v.x = p.x + dist * xderection;

    if (yderection > 0)
    {
      if (image.rows - p.y < 128)
      {
        dist = image.rows - p.y;
      }
    }
    else
    {
      if (p.y < 128)
      {
        dist = p.y;
      }
    }
    v.y = p.y + dist * yderection;*/

    /*v.x = rng.uniform(-255.0, 255.0);
    v.y = rng.uniform(-255.0, 255.0);*/
    v.x = rng.uniform(-128.0, 128.0);
    v.y = rng.uniform(-128.0, 128.0);

    cout << "v.x" << v.x << endl;
    cout << "v.y:" << v.y << endl;
    cout << "sqrt:" << sqrt((p.x - v.x)*(p.x - v.x) + (p.y - v.y) * (p.y - v.y)) << endl;

    //3.生成w
    //3.1归一化后的每个顶点到pv直线的距离
    Mat dmat(image.size(), CV_32FC1, Scalar::all(0));

    for (int j = 0; j < dmat.rows; j++)
    {
      for (int i = 0; i < dmat.cols; i++)
      {
        float distance = getDist_P2L(Point(i, j), p, v);
        // cout << "distance:" << distance << endl;
        dmat.at<float>(j, i) = distance;
      }
    }
    normalize(dmat, dmat, 1, 0, CV_MINMAX);

    //v = 0.5 * v;

    float a1 = rng.uniform(0.1, 1.0);
    cout << "a:" << a1 << endl;
    generate_flod(image, dmat, v, a1, idx, savepath, timename);
    idx++;
    generate_curves(image, dmat, v, a1, idx, savepath, timename);
    idx++;

    float a2 = rng.uniform(1.0, 9.0);
    cout << "a:" << a2 << endl;
    generate_flod(image, dmat, v, a2, idx, savepath, timename);
    idx++;
    generate_curves(image, dmat, v, a2, idx, savepath, timename);
    idx++;
  }
}



int main()
{
  string opendir = "pic/";
  string savepath = "d:/deformation/";

  vector<string> vecsuffix;
  vecsuffix.push_back("jpg");
  vecsuffix.push_back("bmp");
  vecsuffix.push_back("png");
  vecsuffix.push_back("jpeg");
  vector<string> vecfiles;
  string strprefix;
  get_files(opendir, vecsuffix, vecfiles, strprefix);

  for (int i = 0; i < vecfiles.size(); i++)
  {
    cout << "processing image:" << opendir + vecfiles[i] << endl;
    Mat image = imread(opendir + vecfiles[i]);
    struct tm *local;
    time_t t;
    t = time(NULL);
    local = localtime(&t);
    string timename = to_string(local->tm_year) + to_string(local->tm_mon) + to_string(local->tm_mday) + to_string(local->tm_hour) + to_string(local->tm_min) + to_string(local->tm_sec);
    if (imwrite(savepath + "src" + timename + ".jpg", image))
    {
      execute_deformation(image, timename, savepath);
    }
  }

  return 0;
}