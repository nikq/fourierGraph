

/// cvdft.h  ::  core of fourierGraph analysis.
/// copyright 2015 by Hajime UCHIMURA.
/// please contact before business use.


#define FLAG_NOAXIS 1
#define FLAG_NOLINE 2
#define FLAG_BW     4 // 白黒.
#define FLAG_MORELINE 8

namespace FOURIERGRAPH
{

  // LogTransformは両対数グラフ用の座標変換です.
  // FLOATDRAWはアンチエイリアスラインの描画ライブラリです.
  
  class ImageDFT {
  public:

    typedef struct{
      float value_, weight_;
      float min_, max_;
    } BinEntry;
    typedef std::vector< BinEntry > BinList;
    typedef std::vector< BinList  > BinArray;

  public:

    BinArray          spectrum_; // 各ミップのスペクトラム.
    bool              is_rgb_;
    bool              is_mipmap_;
    int               mipmap_level_;
    int               color_channels_;

    ImageDFT(){
      init();
    }
    ~ImageDFT(){
      ;
    }

    inline float minf( float a, float b ){ return (a>b)?b:a; }
    inline float maxf( float a, float b ){ return (a<b)?b:a; }

    // 横上並びミップマップのROIを返す.
    int getMipMAX( const cv::Mat& src ){
      return __lzcnt( (src.cols+1)/2 );
    }
    cv::Mat getMipROI( const cv::Mat& src, int miplv ){
      int w = (src.cols + 1) / 2;
      int h =  src.rows;
      int x = 0;
      //int y = 0;
      for(int i=0;i<miplv;i++){
        x += w;
        w /= 2;
        h /= 2;
      }
      return cv::Mat( src, cv::Rect( x, 0, w, h ) );
    }

    int doDFT( const cv::Mat& src, cv::Mat& dest ) {

      cv::Size s_size = src.size();

      cv::Mat complex_image;
      cv::Mat real_image;

      cv::copyMakeBorder(
        src,
        real_image,
        0, cv::getOptimalDFTSize( src.rows ) - src.rows,
        0, cv::getOptimalDFTSize( src.cols ) - src.cols,
        cv::BORDER_CONSTANT, cv::Scalar::all(0) );

      {
        cv::Mat planes[] = { cv::Mat_<float>( real_image ), cv::Mat::zeros(real_image.size(),CV_32F) } ;
        cv::merge( planes, 2, complex_image );
      }

      cv::Mat dft_image;
      cv::dft( complex_image, dft_image, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT );
      dest = cv::Mat::zeros( src.size(), CV_32F );

      {
        cv::Mat planes[2];
        cv::split( dft_image, planes );
        cv::magnitude( planes[0], planes[1], dest ); // dest = sqrt( planes[0]^2 + planes[1] ^2 ) ;

        {
          cv::Mat tmp;
          int cx = dest.cols/2;
          int cy = dest.rows/2;
          for(int i=0; i<=cx; i+=cx) {
            cv::Mat qs( dest, cv::Rect(i^cx, 0,cx,cy));
            cv::Mat qd( dest, cv::Rect(i   ,cy,cx,cy));
            qs.copyTo(tmp);
            qd.copyTo(qs);
            tmp.copyTo(qd);
          }
        }
      }

      return 0;
    }

    inline float abs_norm( float x, float y, float l ){
      return sqrtf( x * x + y * y ) / l;
    }

    void init( bool is_rgb = false, bool is_mipmap = false ) {
      spectrum_.clear();
      is_rgb_    = is_rgb;
      is_mipmap_ = is_mipmap;
      mipmap_level_   = 0;
      color_channels_ = 0;
    }

    void spectrum( const cv::Mat& src, BinList &bin, float scale ) {
      cv::Size s_size = src.size();
      int  src_cols = s_size.width;
      int  src_rows = s_size.height;
      int  l = ((src_cols < src_rows) ? src_cols : src_rows)/2;

      float cx = (float)src_cols / 2.f;
      float cy = (float)src_rows / 2.f;
      float cl = (cx<cy) ? cx:cy;

      bin.clear();
      bin.resize( l );

      for(int i=0;i<l;i++){
        bin[i].value_ = bin[i].weight_ = 0.f;
        bin[i].min_ = FLT_MAX;
        bin[i].max_ = FLT_MIN;
      }

      for(int y=0;y<src_rows;y++){
        for(int x=0;x<src_cols;x++){

          float r = abs_norm( (float)x - cx, (float)y - cy, cl );
          float value = src.at<float>(y,x) * scale;

          int bin_index = (int)( l * r );
          if( bin_index >= bin.size() )
            continue;

          bin[bin_index].value_  += value;
          bin[bin_index].weight_ += 1.f;
          bin[bin_index].min_     = minf( bin[bin_index].min_, value );
          bin[bin_index].max_     = maxf( bin[bin_index].max_, value );
        }
      }
    }

    int analyze( unsigned char *image, int width, int height, bool is_rgb, bool is_mipmap ) {
      cv::Mat src = cv::Mat( height, width, CV_8UC4, image ).clone();
      // QImageからはRGBストア, OpenCVはBGRストアだけど、グレイにしちゃうから関係なし.
      // グラフもRGBで書くから問題なし.

      if( !src.data )
        return 1;

      init(is_rgb,is_mipmap);

      if( !is_rgb ){
        // グレイにする.
        cv::cvtColor( src, src, CV_BGRA2GRAY );
        //cv::imwrite( "e:\\test.png", src );
      }

      cv::Mat rgb[3];
      cv::split( src, rgb );

      int maxlv = 1;
      if( is_mipmap ){
        maxlv = getMipMAX( rgb[0] );
        maxlv -= 2; // 上位ミップは小さすぎて分析できない.
      }
      mipmap_level_   = maxlv;
      color_channels_ = is_rgb ? 3 : 1;

      for(int lv = 0; lv < mipmap_level_; lv ++ ) {
        for(int ch = 0; ch < color_channels_; ch ++ ) {

          cv::Mat dst;
          cv::Mat roi;

          if( is_mipmap )
            roi = getMipROI( rgb[ch], lv );
          else
            roi = rgb[ch];

          doDFT( roi, dst );

          double min,max;
          minMaxLoc( dst, &min, &max );

          BinList bin;
          spectrum( dst, bin, 1e5 / max );
          spectrum_.push_back( bin );
        }
      }

      return 0;
    }


    void drawAxis( FLOATDRAW::Canvas &canvas ) {

      float width = canvas._width;
      float height= canvas._height;
      LogTransform plot(width,height);

      // draw base axis.
      FLOATDRAW::Color axis = FLOATDRAW::Color(0.25f,0.25f,0.25f);

      float step_x = pow(  2.f, 1.0f );
      float step_y = pow( 10.f, 0.5f );
      for( float x = plot.x_lo_ ; x < plot.x_hi_ ; x *= step_x ){
        float lx, ly;
        plot.transform( x, 1.f, lx, ly );
        canvas.drawLine( lx, 0.f, lx, height, axis, 1.f, 1.f );
      }
      for( float y = 1e-2f; y < 1e5f ; y *= step_y ){
        float lx, ly;
        plot.transform( 1.f, y, lx, ly );
        canvas.drawLine( 0.f, ly, width, ly, axis, 1.f, 1.f );
      }
    }

    void drawLine( FLOATDRAW::Canvas &canvas , int drawBase, float baseValue ) {

      float width = canvas._width;
      float height= canvas._height;
      LogTransform plot(width,height);

      // draw 1/f line
      FLOATDRAW::Color inv_f = FLOATDRAW::Color(0.25f,0.25f,0.25f);
      float s = 2.f;
      float g = 1.f;
      float a = 0.5f;

      if( drawBase > 1 ) {
        inv_f = FLOATDRAW::Color(0.75f,0.5f,0.25f);
        s = 4.f;
        g = 1.f;
        a = 1.f;
      }

      float step_x = pow(  2.f, 1.0f );
      float step_y = pow( 10.f, 0.5f );
      float y = baseValue;
      if( y > 0.f ){
        for( float x = plot.x_lo_ ; x < plot.x_hi_ ; x *= step_x ) {
          float x1,x2,y1,y2;
          plot.transform( x          , y /   x           , x1, y1 );
          plot.transform( x * step_x , y / ( x * step_x ), x2, y2 );
          canvas.drawLine(x1,y1,x2,y2,inv_f,s,g,a);
        }
      }
    }

    void drawGraph( FLOATDRAW::Canvas &canvas, BinArray & array, int flags = 0 ) {

      float width = canvas._width;
      float height= canvas._height;
      LogTransform plot(width,height);

      if( flags & FLAG_BW )
        canvas.fill( FLOATDRAW::Color( 0,0,0 ) );
      else
        canvas.fill( FLOATDRAW::Color( 0.125f,0.125f,0.125f) );

      if( !(flags & FLAG_NOAXIS) )
        drawAxis( canvas );
      if( array.size() > 0 && !(flags & FLAG_NOLINE ) )
        drawLine( canvas, (flags&FLAG_MORELINE) ? 2 : 1, (array[0][1].value_ / array[0][1].weight_)/10.f );

      for(int lv = 0; lv < mipmap_level_; lv ++ ) {
        for(int ch = 0; ch < color_channels_; ch ++ ) {

          BinList &list( array[ lv * color_channels_ + ch ] );

          float a = (float)(lv + 1) / mipmap_level_;

          FLOATDRAW::Color color;
          float alpha;

          if( !is_rgb_ ){
            color = is_mipmap_ ? FLOATDRAW::Color( a, 1.f - a, a ) : FLOATDRAW::Color( 1.f,1.f,1.f );
            alpha = 1.f;
          } else {
            color = FLOATDRAW::Color(
              ch == 0 ? 0.f : 1.f ,
              ch == 1 ? 0.f : 1.f ,
              ch == 2 ? 0.f : 1.f );
            alpha = a;
          }

          for( int i = 1 ; i < list.size() - 1 ; i++ ) {

            int i1 = i;
            int i2 = i+1;

            float linear_x1 = i1;
            float linear_x2 = i2;
            float linear_y1 = (list[i1].weight_ > 0.f) ? list[i1].value_ / list[i1].weight_ : 0.f;
            float linear_y2 = (list[i2].weight_ > 0.f) ? list[i2].value_ / list[i2].weight_ : 0.f;

            float x1,x2,y1,y2;

            plot.transform( linear_x1, linear_y1, x1, y1 );
            plot.transform( linear_x2, linear_y2, x2, y2 );
            canvas.drawLine(x1,y1,x2,y2,color,1.f,1.f,alpha);
          }
        }
        }
    }
    
    
    // 分析の終わったグラフを書き出すインターフェース
    void draw( FLOATDRAW::Canvas &canvas ) {
      drawGraph(canvas,spectrum_);
    }
    void draw( unsigned char *rgb, int width, int height ) {
      FLOATDRAW::Canvas canvas;
      canvas.setup(width,height,1.f);
      drawGraph(canvas,spectrum_);
      canvas.tonemap24(rgb);
    }
    void render( unsigned char *rgb, int width, int height ){
      FLOATDRAW::Canvas canvas;
      canvas.setup(width,height,1.f);
      drawGraph(canvas,spectrum_,FLAG_BW|FLAG_MORELINE);
      canvas.tonemap24(rgb);
    }
  };
}

