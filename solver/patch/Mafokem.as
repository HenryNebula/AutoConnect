package kawai2_fla
{
   import adobe.utils.*;
   import fl.motion.Color;
   import flash.accessibility.*;
   import flash.display.*;
   import flash.errors.*;
   import flash.events.*;
   import flash.external.*;
   import flash.filters.*;
   import flash.geom.*;
   import flash.media.*;
   import flash.net.*;
   import flash.printing.*;
   import flash.profiler.*;
   import flash.sampler.*;
   import flash.system.*;
   import flash.text.*;
   import flash.ui.*;
   import flash.utils.*;
   import flash.xml.*;
   import kokysyqes.*;
   import neruwelof.*;
   
   public dynamic class Mafokem extends MovieClip
   {
      
      public var txt_score:TextField;
      
      public var mytimeinter:uint;
      
      public var tywu:uint;
      
      public var sirivyb:int;
      
      public var quzebyror:*;
      
      public var fofih:MovieClip;
      
      public var gameTimer1:Timer;
      
      public var senuheq:Timer;
      
      public var rejorij:MovieClip;
      
      public var timebar:MovieClip;
      
      public var homeurl:String;
      
      public var btnrestart:SimpleButton;
      
      public var txt_xipaishu:TextField;
      
      public var score_jiang:int;
      
      public var score_chushi:int;
      
      public var chongzi:MovieClip;
      
      public var geturl:String;
      
      public var pec:String;
      
      public var timeoutId:int;
      
      public var laqop:uint;
      
      public var pemo:uint;
      
      public var siqufibi:MovieClip;
      
      public var myContinue:MovieClip;
      
      public var intduishu:int;
      
      public var wec:SoundChannel;
      
      public var zyvonyg:uint;
      
      public var gameid:int;
      
      public var score:int;
      
      public var lianjidanwei:uint;
      
      public var hijo:uint;
      
      public var beijing:MovieClip;
      
      public var time:int;
      
      public var intjihui:int;
      
      public var posturl:String;
      
      public var shibaiyuanyin:String;
      
      public var lianjieshu:int;
      
      public var vum:Lili;
      
      public var mc_show2:MovieClip;
      
      public var mc_show3:MovieClip;
      
      public var mc_show5:MovieClip;
      
      public var mc_show6:MovieClip;
      
      public var txt_level:TextField;
      
      public var mc_show1:MovieClip;
      
      public var score_shijianfen:int;
      
      public var mc_show4:MovieClip;
      
      public var neg:Sprite;
      
      public var _4399_function_gameList_id:String;
      
      public var stotaltime:int;
      
      public var nar:Color;
      
      public var lianjicishu:uint;
      
      public var btnpause:SimpleButton;
      
      public var myLHXY:Array;
      
      public var btnxipai:SimpleButton;
      
      public var maxtotaltime:int;
      
      public var _4399_function_score_id:String;
      
      public var txt_tishishu:TextField;
      
      public var _4399_function_ad_id:String;
      
      public var mebyhuham:MovieClip;
      
      public var myone:MovieClip;
      
      public var btnhome:SimpleButton;
      
      public var txt_time:TextField;
      
      public var kepigi:uint;
      
      public var btntishi:SimpleButton;
      
      public var woleriwom:uint;
      
      public var zaqu:int;
      
      public var txt_xingyun:TextField;
      
      public var tuwupi:Date;
      
      public var hifema:Array;
      
      public var txt_leveltoptext:TextField;
      
      public var kef:uint;
      
      public var zuho:uint;
      
      public var score_link:int;
      
      public var levelscore:int;
      
      public var lasinus:uint;
      
      public var saviwoq:Qawecefo;
      
      public var score_guoguan:int;
      
      public var score_shijian:int;
      
      public var faraquno:Boolean;
      
      public var lianjifen:uint;
      
      public var timescore:int;
      
      public var jewyr:Number;

      public var kimubo:Number;

      // ---- AutoConnect autonomous solver + ExternalInterface bridge ----
      public var acTimer:Timer;
      public var acClears:uint;
      public var acFails:uint;
      public var acEnabled:Boolean;
      public var acInterval:uint;
      public var acLastFail:String;

      public function Mafokem()
      {
         super();
         addFrameScript(0,frame1);
      }
      
      public function girod(param1:Array) : void
      {
         var _loc4_:MovieClip = null;
         var _loc2_:int = int(param1.length);
         var _loc3_:int = int(param1[0].length);
         while(mebyhuham.numChildren > 0)
         {
            _loc4_ = mebyhuham.getChildAt(0) as MovieClip;
            mebyhuham.removeChild(_loc4_);
         }
      }
      
      public function _mapleft(param1:Array) : Array
      {
         var _loc4_:Array = null;
         var _loc5_:int = 0;
         var _loc6_:* = 0;
         var _loc7_:uint = 0;
         var _loc2_:Array = fanu(param1);
         var _loc3_:int = 0;
         while(_loc3_ < _loc2_.length)
         {
            _loc4_ = new Array();
            _loc5_ = 0;
            while(_loc5_ < _loc2_[0].length)
            {
               _loc4_.push(_loc2_[_loc3_][_loc5_].type);
               _loc5_++;
            }
            _loc6_ = int(_loc4_.length - 1);
            while(_loc6_ >= 0)
            {
               if(_loc4_[_loc6_] == -1)
               {
                  _loc4_.splice(_loc6_,1);
                  _loc4_.push(-1);
               }
               _loc6_--;
            }
            _loc4_.pop();
            _loc4_.unshift(-1);
            _loc7_ = 0;
            while(_loc7_ < _loc4_.length)
            {
               _loc2_[_loc3_][_loc7_].type = _loc4_[_loc7_];
               _loc7_++;
            }
            _loc3_++;
         }
         return fanu(_loc2_);
      }
      
      public function addone() : void
      {
         gameTimer1.stop();
         chongzhi();
         txt_time.text = String(stotaltime);
         addChild(myone);
         myone.x = 0;
         myone.y = 0;
      }
      
      public function fgameover() : void
      {
         girod(vum.nidi);
         score_shijian = 0;
         score_shijianfen = 0;
         var _loc1_:Sahumu = new Sahumu();
         var _loc2_:SoundChannel = _loc1_.play(0);
         siqufibi = new Gal();
         addChild(siqufibi);
         siqufibi.x = 0;
         siqufibi.y = 0;
         siqufibi.btnback.visible = true;
         siqufibi.btnok.visible = !siqufibi.btnback.visible;
         paihang(score);
      }
      
      public function NextLevel() : void
      {
         var _loc1_:* = undefined;
         hifema.length = 0;
         score_link = 0;
         score_chushi = score;
         score_jiang = 0;
         score_shijian = 0;
         score_shijianfen = 0;
         score_guoguan = 0;
         intjihui = myLHXY[zuho - 1][7];
         levelscore = myLHXY[zuho - 1][9];
         timescore = myLHXY[zuho - 1][10];
         lianjidanwei = myLHXY[zuho - 1][11];
         lianjifen = 0;
         lianjicishu = 0;
         txt_xingyun.text = "";
         lianjieshu = 0;
         nar.brightness = -0.7;
         beijing.transform.colorTransform = nar;
         if(lasinus >= kef)
         {
            lasinus = kef;
         }
         if(zyvonyg >= laqop)
         {
            zyvonyg = laqop;
         }
         txt_xipaishu.text = String(lasinus);
         txt_tishishu.text = String(zyvonyg);
         if(rejorij)
         {
            rejorij.light.visible = false;
            rejorij = null;
         }
         if(fofih)
         {
            fofih.light.visible = false;
            fofih = null;
         }
         tywu = myLHXY[zuho - 1][1];
         pemo = myLHXY[zuho - 1][2];
         jewyr = myLHXY[zuho - 1][3];
         kimubo = myLHXY[zuho - 1][4];
         sirivyb = (tywu - 2) / 2;
         zaqu = (pemo - 2) / 2;
         intduishu = (tywu - 2) * (pemo - 2) / 2;
         maxtotaltime = myLHXY[zuho - 1][6];
         txt_leveltoptext.text = "第 " + zuho + " 关:" + " " + myLHXY[zuho - 1][0] + " " + String(myLHXY[zuho - 1][5]) + "种" + pec + " " + maxtotaltime + "秒";
         do
         {
            _loc1_ = vaqike(tywu,pemo,myLHXY[zuho - 1][5]);
            vum.ruzokyn = _loc1_;
         }
         while(!lujyl());
         fyfam(_loc1_);
         txt_level.text = String(zuho);
         stotaltime = myLHXY[zuho - 1][6];
         gameTimer1.start();
         gameTimer1.addEventListener(TimerEvent.TIMER,jil);
         stage.addEventListener(MouseEvent.MOUSE_DOWN,daqal);
         timebar.width = stotaltime / maxtotaltime * 160;
         chongzi.x = timebar.x + timebar.width - 10;
         zikuvi();
         if(faraquno)
         {
            gijobufi();
         }
      }
      
      public function chongzhi() : *
      {
         hifema.length = 0;
         score_link = 0;
         score_chushi = 0;
         score_guoguan = 0;
         score_jiang = 0;
         score_shijian = 0;
         score_shijianfen = 0;
         lianjidanwei = 0;
         lianjifen = 0;
         lianjicishu = 0;
         score = 0;
         shibaiyuanyin = "";
         lianjieshu = 0;
         stotaltime = 0;
         time = 0;
         pec = "宠物";
         kef = 10;
         laqop = 10;
         lasinus = 5;
         zyvonyg = 5;
         woleriwom = 39;
         kepigi = 39;
         zuho = 1;
         myLHXY = [["不动",14,10,-2,0,12,90,50,100,10,10,10],["下移",14,10,-2,0,14,100,100,100,10,10,10],["左移",14,10,-2,0,16,110,125,100,10,10,10],["上移",14,10,-2,0,18,120,150,100,10,10,10],["右移",14,10,-2,0,20,130,175,100,10,10,10],["上下分离",14,10,-2,0,22,140,200,200,20,20,20],["上下靠拢",14,10,-2,0,24,150,225,200,20,20,20],["左右分离",14,10,-2,0,26,200,250,200,20,20,20],["左右靠拢",14,10,-2,0,28,210,275,200,20,20,20],["横向错位",14,10,-2,0,30,220,300,200,20,20,20],["四周分散",14,10,-2,0,34,230,100,500,50,50,50],["纵向错位",14,10,-2,0,38,240,125,500,50,50,50],["中心靠拢",14,10,-2,0,42,250,150,1000,100,100,100]];
         hijo = myLHXY.length;
         intjihui = myLHXY[zuho - 1][7];
         levelscore = myLHXY[zuho - 1][9];
         timescore = myLHXY[zuho - 1][10];
         lianjidanwei = myLHXY[zuho - 1][11];
         btnxipai.mouseEnabled = true;
         btntishi.mouseEnabled = true;
         btnpause.mouseEnabled = true;
         btnhome.mouseEnabled = true;
         btnrestart.mouseEnabled = true;
         txt_score.text = String(score);
         txt_xipaishu.text = String(lasinus);
         txt_tishishu.text = String(zyvonyg);
      }
      
      public function setHold(param1:*) : void
      {
         root["serviceHold"] = param1;
      }
      
      public function zymuhucyr() : *
      {
         var _loc1_:SoundChannel = null;
         var _loc2_:Myqemefi = new Myqemefi();
         _loc1_ = _loc2_.play(0);
      }
      
      public function lujyl() : Boolean
      {
         var _loc1_:Array = vum.cunufi();
         if(_loc1_)
         {
            return true;
         }
         return false;
      }
      
      public function createTiShi(param1:MouseEvent) : void
      {
         if(zyvonyg > 0)
         {
            if(rejorij)
            {
               rejorij.light.visible = false;
               rejorij = null;
            }
            if(fofih)
            {
               fofih.light.visible = false;
               fofih = null;
            }
            if(cokuty() == true)
            {
               --zyvonyg;
               txt_tishishu.text = String(zyvonyg);
               if(zyvonyg <= 0)
               {
                  btntishi.mouseEnabled = false;
               }
               else
               {
                  btntishi.mouseEnabled = true;
               }
               sicikol();
            }
            else if(lasinus <= 0)
            {
               shibaiyuanyin = "生命耗尽，无牌可连";
               gameTimer1.stop();
               fgameover();
            }
            else
            {
               createNewMap(param1);
            }
         }
      }
      
      public function zikuvi() : *
      {
         var _loc1_:SoundChannel = null;
         var _loc2_:Qup = new Qup();
         _loc1_ = _loc2_.play(0);
      }
      
      public function createTBKMap() : void
      {
         var _loc1_:Array = hek(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function createLRMap() : void
      {
         var _loc1_:Array = zefyk(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function bojekot() : *
      {
         var _loc1_:SoundChannel = null;
         var _loc2_:Lypobef = new Lypobef();
         _loc1_ = _loc2_.play(0);
      }
      
      internal function frame1() : *
      {
         stop();
         if(Security.sandboxType != "application")
         {
            Security.allowDomain("*");
         }
         _4399_function_score_id = "d8c8d4731a33a0a581edc746e73eadc7200";
         _4399_function_ad_id = "92d6cef76cd06829e088932fe9fd819b";
         _4399_function_gameList_id = "944c23f5e64a80647f8d0f3435f5c7a8";
         faraquno = false;
         mytimeinter = 300;
         gameid = 7761;
         homeurl = "";
         posturl = "";
         geturl = "";
         shibaiyuanyin = "";
         nar = new Color();
         saviwoq = new Qawecefo();
         vum = new Lili();
         myContinue = new thisContinue();
         myone = new one();
         mebyhuham = new MovieClip();
         gameTimer1 = new Timer(1000);
         hifema = new Array();
         mc_show1 = new MovieClip();
         mc_show2 = new MovieClip();
         mc_show3 = new MovieClip();
         mc_show4 = new MovieClip();
         mc_show5 = new MovieClip();
         mc_show6 = new MovieClip();
         neg = new Sprite();
         addChild(neg);
         addChild(mebyhuham);
         // AutoConnect: skip the title overlay and jump straight into level 1,
         // then start the autonomous solver + register ExternalInterface hooks.
         chongzhi();
         NextLevel();
         acInstallSolver();
         btntishi.addEventListener(MouseEvent.CLICK,createTiShi);
         btnxipai.addEventListener(MouseEvent.CLICK,createNewMap);
         btnrestart.addEventListener(MouseEvent.CLICK,createMap);
         btnpause.addEventListener(MouseEvent.CLICK,fContinue);
         btnhome.addEventListener(MouseEvent.CLICK,openURL);
         senuheq = new Timer(5);
      }
      
      public function cokuty() : Boolean
      {
         var _loc1_:Array = vum.cunufi();
         if(_loc1_)
         {
            (mebyhuham.getChildByName("myicon_x" + _loc1_[0][0] + "y" + _loc1_[0][1]) as MovieClip).shine.visible = true;
            (mebyhuham.getChildByName("myicon_x" + _loc1_[1][0] + "y" + _loc1_[1][1]) as MovieClip).shine.visible = true;
            return true;
         }
         return false;
      }
      
      public function dongzuo() : void
      {
         clearTimeout(timeoutId);
         switch(zuho)
         {
            case 1:
               break;
            case 2:
               createBottomMap();
               break;
            case 3:
               createLeftMap();
               break;
            case 4:
               createTopMap();
               break;
            case 5:
               createRightMap();
               break;
            case 6:
               createTBMap();
               break;
            case 7:
               createTBKMap();
               break;
            case 8:
               createLRMap();
               break;
            case 9:
               createLRKMap();
               break;
            case 10:
               createULDRMap();
               break;
            case 11:
               createTBMap();
               createLRMap();
               break;
            case 12:
               createLURDMap();
               break;
            case 13:
               createLRKMap();
               createTBKMap();
         }
         if(!lujyl())
         {
            createNewMap(null);
         }
      }
      
      public function createTBMap() : void
      {
         var _loc1_:Array = hicovopum(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function nidi(param1:Array) : Array
      {
         var _loc9_:uint = 0;
         var _loc10_:uint = 0;
         var _loc2_:Array = param1;
         var _loc3_:Array = new Array();
         var _loc4_:int = int(param1.length);
         var _loc5_:int = int(param1[0].length);
         var _loc6_:uint = 0;
         while(_loc6_ < _loc4_)
         {
            _loc9_ = 0;
            while(_loc9_ < _loc5_)
            {
               if(_loc2_[_loc6_][_loc9_].type != -1)
               {
                  _loc3_.push(_loc2_[_loc6_][_loc9_].type);
               }
               _loc9_++;
            }
            _loc6_++;
         }
         _loc3_.sort(randomsort);
         var _loc7_:int = 0;
         var _loc8_:uint = 0;
         while(_loc8_ < _loc4_)
         {
            _loc10_ = 0;
            while(_loc10_ < _loc5_)
            {
               if(_loc2_[_loc8_][_loc10_].type != -1)
               {
                  _loc2_[_loc8_][_loc10_].type = _loc3_[_loc7_];
                  _loc7_++;
               }
               _loc10_++;
            }
            _loc8_++;
         }
         return _loc2_;
      }
      
      public function addone2() : void
      {
         gameTimer1.stop();
         chongzhi();
         txt_time.text = String(stotaltime);
         var _loc1_:MovieClip = new one();
         addChild(_loc1_);
         _loc1_.x = 0;
         _loc1_.y = 0;
         var _loc2_:MovieClip = new showtop();
         addChild(_loc2_);
         _loc2_.x = 0;
         _loc2_.y = 0;
      }
      
      public function pocufy() : *
      {
         var _loc1_:SoundChannel = null;
         var _loc2_:Dyveze = new Dyveze();
         _loc1_ = _loc2_.play(0);
      }
      
      public function createRightMap() : void
      {
         var _loc1_:Array = _mapright(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function roviga(param1:Array) : Boolean
      {
         var _loc5_:int = 0;
         var _loc2_:Boolean = true;
         var _loc3_:Array = param1;
         var _loc4_:int = 0;
         while(_loc4_ < _loc3_.length)
         {
            _loc5_ = 0;
            while(_loc5_ < _loc3_[0].length)
            {
               if(_loc3_[_loc4_][_loc5_].type != -1)
               {
                  _loc2_ = false;
                  break;
               }
               _loc5_++;
            }
            _loc4_++;
         }
         return _loc2_;
      }
      
      public function createMap(param1:MouseEvent) : void
      {
         addone();
      }
      
      public function random(param1:int, param2:int) : int
      {
         if(param1 == param2)
         {
            return param1;
         }
         if(param1 > param2)
         {
            return param2 + int(Math.random() * (param1 - param2 + 1));
         }
         return param1 + int(Math.random() * (param2 - param1 + 1));
      }
      
      public function openURL(param1:MouseEvent) : void
      {
         var _loc2_:* = root["serviceHold"];
         if(_loc2_)
         {
            _loc2_.showGameList();
         }
      }
      
      public function sicikol() : *
      {
         var _loc1_:SoundChannel = null;
         var _loc2_:Saqicila = new Saqicila();
         _loc1_ = _loc2_.play(0);
      }
      
      public function randomsort(param1:Object, param2:Object) : int
      {
         return Math.pow(-1,Math.floor(Math.random() * 2));
      }
      
      public function zefyk(param1:Array) : Array
      {
         var _loc4_:Array = null;
         var _loc5_:Array = null;
         var _loc6_:int = 0;
         var _loc7_:Array = null;
         var _loc8_:int = 0;
         var _loc9_:* = 0;
         var _loc10_:uint = 0;
         var _loc2_:Array = fanu(param1);
         var _loc3_:int = 0;
         while(_loc3_ < _loc2_.length)
         {
            _loc4_ = new Array();
            _loc5_ = new Array();
            _loc6_ = 0;
            while(_loc6_ <= sirivyb)
            {
               _loc5_.push(_loc2_[_loc3_][_loc6_].type);
               _loc6_++;
            }
            _loc7_ = new Array();
            _loc8_ = sirivyb + 1;
            while(_loc8_ < _loc2_[0].length)
            {
               _loc7_.push(_loc2_[_loc3_][_loc8_].type);
               _loc8_++;
            }
            _loc9_ = 0;
            while(_loc9_ <= sirivyb)
            {
               if(_loc7_[_loc9_] == -1)
               {
                  _loc7_.splice(_loc9_,1);
                  _loc7_.unshift(-1);
               }
               _loc9_++;
            }
            _loc7_.push(-1);
            _loc7_.shift();
            _loc9_ = sirivyb;
            while(_loc9_ >= 0)
            {
               if(_loc5_[_loc9_] == -1)
               {
                  _loc5_.splice(_loc9_,1);
                  _loc5_.push(-1);
               }
               _loc9_--;
            }
            _loc5_.pop();
            _loc5_.unshift(-1);
            _loc4_ = _loc5_.concat(_loc7_);
            _loc10_ = 0;
            while(_loc10_ < _loc4_.length)
            {
               _loc2_[_loc3_][_loc10_].type = _loc4_[_loc10_];
               _loc10_++;
            }
            _loc3_++;
         }
         return fanu(_loc2_);
      }
      
      public function getIconArr(param1:uint, param2:uint, param3:uint) : Array
      {
         var _loc4_:int = 0;
         var _loc5_:Array = new Array();
         var _loc6_:uint = (param1 - 2) * (param2 - 2);
         var _loc7_:uint = 0;
         while(_loc7_ < _loc6_)
         {
            _loc5_.push(_loc7_ % param3);
            _loc4_++;
            _loc5_.push(_loc7_ % param3);
            if(++_loc4_ == _loc6_)
            {
               break;
            }
            _loc7_++;
         }
         return _loc5_;
      }
      
      public function paihang(param1:uint) : void
      {
         var _loc2_:* = undefined;
         if(param1 >= 100)
         {
            _loc2_ = root["serviceHold"];
            if(_loc2_)
            {
               _loc2_.showRefer(param1);
            }
         }
      }
      
      public function _mapbottom(param1:Array) : Array
      {
         var _loc4_:Array = null;
         var _loc5_:int = 0;
         var _loc6_:int = 0;
         var _loc7_:uint = 0;
         var _loc2_:Array = param1;
         var _loc3_:int = 0;
         while(_loc3_ < _loc2_.length)
         {
            _loc4_ = new Array();
            _loc5_ = 0;
            while(_loc5_ < _loc2_[0].length)
            {
               _loc4_.push(_loc2_[_loc3_][_loc5_].type);
               _loc5_++;
            }
            _loc6_ = 0;
            while(_loc6_ <= _loc4_.length - 1)
            {
               if(_loc4_[_loc6_] == -1)
               {
                  _loc4_.splice(_loc6_,1);
                  _loc4_.unshift(-1);
               }
               _loc6_++;
            }
            _loc4_.shift();
            _loc4_.push(-1);
            _loc7_ = 0;
            while(_loc7_ < _loc4_.length - 1)
            {
               _loc2_[_loc3_][_loc7_].type = _loc4_[_loc7_];
               _loc7_++;
            }
            _loc3_++;
         }
         return _loc2_;
      }
      
      public function fanu(param1:Array) : Array
      {
         var _loc5_:Array = null;
         var _loc6_:uint = 0;
         var _loc2_:Array = param1;
         var _loc3_:Array = new Array();
         var _loc4_:uint = 0;
         while(_loc4_ < _loc2_[0].length)
         {
            _loc5_ = new Array();
            _loc6_ = 0;
            while(_loc6_ < _loc2_.length)
            {
               _loc5_.push(_loc2_[_loc6_][_loc4_]);
               _loc6_++;
            }
            _loc3_.push(_loc5_);
            _loc4_++;
         }
         return _loc3_;
      }
      
      public function addlevelindex() : void
      {
         huanmandongzuo();
      }
      
      public function fContinue(param1:MouseEvent) : void
      {
         if(rejorij)
         {
            rejorij.light.visible = false;
            rejorij = null;
         }
         if(fofih)
         {
            fofih.light.visible = false;
            fofih = null;
         }
         gameTimer1.stop();
         addChild(myContinue);
         myContinue.x = 0;
         myContinue.y = 0;
      }
      
      public function gijobufi() : void
      {
         quzebyror = setTimeout(gijobufi,mytimeinter);
         if(jigu())
         {
            byqij();
         }
      }
      
      public function createLeftMap() : void
      {
         var _loc1_:Array = _mapleft(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function createULDRMap() : void
      {
         var _loc1_:Array = jyj(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function vaqike(param1:uint, param2:uint, param3:uint) : Array
      {
         var _loc7_:Array = null;
         var _loc8_:uint = 0;
         var _loc9_:int = 0;
         var _loc10_:Object = null;
         var _loc4_:Array = new Array();
         var _loc5_:Array = getIconArr(param1,param2,param3);
         var _loc6_:uint = 0;
         while(_loc6_ < param1)
         {
            _loc7_ = new Array();
            _loc8_ = 0;
            while(_loc8_ < param2)
            {
               if(_loc6_ == 0 || _loc8_ == 0 || _loc6_ == param1 - 1 || _loc8_ == param2 - 1)
               {
                  _loc9_ = -1;
               }
               else
               {
                  _loc9_ = int(_loc5_.splice(random(0,_loc5_.length - 1),1)[0]);
               }
               _loc10_ = {
                  "x":_loc6_,
                  "y":_loc8_,
                  "type":_loc9_
               };
               _loc7_[_loc8_] = _loc10_;
               _loc8_++;
            }
            _loc4_[_loc6_] = _loc7_;
            _loc6_++;
         }
         return _loc4_;
      }
      
      public function createLURDMap() : void
      {
         var _loc1_:Array = boziqy(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function daqal(param1:MouseEvent) : *
      {
         if(param1.target.name.indexOf("myicon_x") < 0)
         {
            if(rejorij)
            {
               rejorij.light.visible = false;
               rejorij = null;
            }
            if(fofih)
            {
               fofih.light.visible = false;
               fofih = null;
            }
            return;
         }
         if(!rejorij)
         {
            rejorij = param1.target as MovieClip;
            rejorij.light.visible = true;
            sicikol();
         }
         else
         {
            fofih = param1.target as MovieClip;
            fofih.light.visible = true;
            if(rejorij == fofih)
            {
               rejorij.light.visible = false;
               fofih.light.visible = false;
               rejorij = fofih = null;
               return;
            }
            byqij();
         }
      }
      
      public function jigu() : Boolean
      {
         var _loc1_:Array = vum.cunufi();
         if(_loc1_)
         {
            (mebyhuham.getChildByName("myicon_x" + _loc1_[0][0] + "y" + _loc1_[0][1]) as MovieClip).shine.visible = true;
            (mebyhuham.getChildByName("myicon_x" + _loc1_[1][0] + "y" + _loc1_[1][1]) as MovieClip).shine.visible = true;
            rejorij = mebyhuham.getChildByName("myicon_x" + _loc1_[0][0] + "y" + _loc1_[0][1]) as MovieClip;
            fofih = mebyhuham.getChildByName("myicon_x" + _loc1_[1][0] + "y" + _loc1_[1][1]) as MovieClip;
            return true;
         }
         return false;
      }
      
      public function huanmandongzuo() : void
      {
         senuheq.start();
         senuheq.addEventListener(TimerEvent.TIMER,mygifu);
      }
      
      public function jyj(param1:Array) : Array
      {
         var _loc5_:Array = null;
         var _loc6_:int = 0;
         var _loc7_:uint = 0;
         var _loc2_:Array = fanu(param1);
         var _loc3_:* = 0;
         var _loc4_:int = 0;
         while(_loc4_ < _loc2_.length)
         {
            _loc5_ = new Array();
            _loc6_ = 0;
            while(_loc6_ < _loc2_[0].length)
            {
               _loc5_.push(_loc2_[_loc4_][_loc6_].type);
               _loc6_++;
            }
            if(_loc4_ <= zaqu)
            {
               _loc3_ = int(_loc5_.length - 1);
               while(_loc3_ >= 0)
               {
                  if(_loc5_[_loc3_] == -1)
                  {
                     _loc5_.splice(_loc3_,1);
                     _loc5_.push(-1);
                  }
                  _loc3_--;
               }
               _loc5_.pop();
               _loc5_.unshift(-1);
            }
            else if(_loc4_ > zaqu)
            {
               _loc3_ = 0;
               while(_loc3_ < _loc5_.length)
               {
                  if(_loc5_[_loc3_] == -1)
                  {
                     _loc5_.splice(_loc3_,1);
                     _loc5_.unshift(-1);
                  }
                  _loc3_++;
               }
               _loc5_.shift();
               _loc5_.push(-1);
            }
            _loc7_ = 0;
            while(_loc7_ < _loc5_.length)
            {
               _loc2_[_loc4_][_loc7_].type = _loc5_[_loc7_];
               _loc7_++;
            }
            _loc4_++;
         }
         return fanu(_loc2_);
      }
      
      public function createLRKMap() : void
      {
         var _loc1_:Array = lezi(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function createBottomMap() : void
      {
         var _loc1_:Array = _mapbottom(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function boziqy(param1:Array) : Array
      {
         var _loc5_:Array = null;
         var _loc6_:int = 0;
         var _loc7_:uint = 0;
         var _loc2_:* = 0;
         var _loc3_:Array = param1;
         var _loc4_:int = 0;
         while(_loc4_ < _loc3_.length)
         {
            _loc5_ = new Array();
            _loc6_ = 0;
            while(_loc6_ < _loc3_[0].length)
            {
               _loc5_.push(_loc3_[_loc4_][_loc6_].type);
               _loc6_++;
            }
            if(_loc4_ <= sirivyb)
            {
               _loc2_ = int(_loc5_.length - 1);
               while(_loc2_ >= 0)
               {
                  if(_loc5_[_loc2_] == -1)
                  {
                     _loc5_.splice(_loc2_,1);
                     _loc5_.push(-1);
                  }
                  _loc2_--;
               }
               _loc5_.pop();
               _loc5_.unshift(-1);
            }
            else if(_loc4_ > sirivyb)
            {
               _loc2_ = 0;
               while(_loc2_ <= _loc5_.length - 1)
               {
                  if(_loc5_[_loc2_] == -1)
                  {
                     _loc5_.splice(_loc2_,1);
                     _loc5_.unshift(-1);
                  }
                  _loc2_++;
               }
               _loc5_.shift();
               _loc5_.push(-1);
            }
            _loc7_ = 0;
            while(_loc7_ < _loc5_.length)
            {
               _loc3_[_loc4_][_loc7_].type = _loc5_[_loc7_];
               _loc7_++;
            }
            _loc4_++;
         }
         return _loc3_;
      }
      
      public function _maptop(param1:Array) : Array
      {
         var _loc4_:Array = null;
         var _loc5_:int = 0;
         var _loc6_:* = 0;
         var _loc7_:uint = 0;
         var _loc2_:Array = param1;
         var _loc3_:int = 0;
         while(_loc3_ < _loc2_.length)
         {
            _loc4_ = new Array();
            _loc5_ = 0;
            while(_loc5_ < _loc2_[0].length)
            {
               _loc4_.push(_loc2_[_loc3_][_loc5_].type);
               _loc5_++;
            }
            _loc6_ = int(_loc4_.length - 1);
            while(_loc6_ >= 0)
            {
               if(_loc4_[_loc6_] == -1)
               {
                  _loc4_.splice(_loc6_,1);
                  _loc4_.push(-1);
               }
               _loc6_--;
            }
            _loc4_.pop();
            _loc4_.unshift(-1);
            _loc7_ = 0;
            while(_loc7_ < _loc4_.length)
            {
               _loc2_[_loc3_][_loc7_].type = _loc4_[_loc7_];
               _loc7_++;
            }
            _loc3_++;
         }
         return _loc2_;
      }
      
      public function _mapright(param1:Array) : Array
      {
         var _loc4_:Array = null;
         var _loc5_:int = 0;
         var _loc6_:int = 0;
         var _loc7_:uint = 0;
         var _loc2_:Array = fanu(param1);
         var _loc3_:int = 0;
         while(_loc3_ < _loc2_.length)
         {
            _loc4_ = new Array();
            _loc5_ = 0;
            while(_loc5_ < _loc2_[0].length)
            {
               _loc4_.push(_loc2_[_loc3_][_loc5_].type);
               _loc5_++;
            }
            _loc6_ = 0;
            while(_loc6_ <= _loc4_.length - 1)
            {
               if(_loc4_[_loc6_] == -1)
               {
                  _loc4_.splice(_loc6_,1);
                  _loc4_.unshift(-1);
               }
               _loc6_++;
            }
            _loc4_.shift();
            _loc4_.push(-1);
            _loc7_ = 0;
            while(_loc7_ < _loc4_.length - 1)
            {
               _loc2_[_loc3_][_loc7_].type = _loc4_[_loc7_];
               _loc7_++;
            }
            _loc3_++;
         }
         return fanu(_loc2_);
      }
      
      public function createNewMap(param1:MouseEvent) : void
      {
         var _loc2_:* = undefined;
         // AutoConnect patch: free reshuffles. The legitimate response to a
         // deadlock is a reshuffle; the arcade life-limit on it is irrelevant
         // to solving the puzzle, so never fail out on a deadlock.
         if(rejorij)
         {
            rejorij.light.visible = false;
            rejorij = null;
         }
         if(fofih)
         {
            fofih.light.visible = false;
            fofih = null;
         }
         if(lasinus > 0)
         {
            --lasinus;
            txt_xipaishu.text = String(lasinus);
            if(lasinus <= 0)
            {
               btnxipai.mouseEnabled = false;
            }
            else
            {
               btnxipai.mouseEnabled = true;
            }
         }
         do
         {
            _loc2_ = nidi(vum.nidi);
            vum.ruzokyn = _loc2_;
         }
         while(!lujyl());
         fyfam(_loc2_);
         zazacuj();
      }
      
      public function jil(param1:TimerEvent) : *
      {
         --stotaltime;
         score_shijian = stotaltime;
         score_shijianfen = score_shijian * timescore;
         if(stotaltime <= 10)
         {
            wec = saviwoq.play(0);
         }
         timebar.width = stotaltime / maxtotaltime * 160;
         chongzi.x = timebar.x + timebar.width - 10;
         txt_time.text = String(stotaltime);
         if(stotaltime <= 0)
         {
            shibaiyuanyin = "时间耗尽";
            gameTimer1.stop();
            fgameover();
         }
      }
      
      public function mygifu(param1:TimerEvent) : *
      {
         var _loc2_:Mutadujub = null;
         var _loc3_:SoundChannel = null;
         --stotaltime;
         if(stotaltime <= 0)
         {
            senuheq.stop();
            _loc2_ = new Mutadujub();
            _loc3_ = _loc2_.play(0);
            siqufibi = new Gal();
            addChild(siqufibi);
            if(zuho <= hijo)
            {
               siqufibi.x = 0;
               siqufibi.y = 0;
               shibaiyuanyin = "恭喜过关!";
               siqufibi.btnback.visible = false;
               siqufibi.btnok.visible = !siqufibi.btnback.visible;
            }
            else
            {
               shibaiyuanyin = "您已通关!";
               siqufibi.btnback.visible = true;
               siqufibi.btnok.visible = !siqufibi.btnback.visible;
               paihang(score);
            }
         }
         else
         {
            score += levelscore;
         }
         txt_score.text = String(score);
         txt_time.text = String(stotaltime);
         timebar.width = stotaltime / maxtotaltime * 160;
         chongzi.x = timebar.x + timebar.width - 10;
      }
      
      public function zazacuj() : *
      {
         var _loc1_:SoundChannel = null;
         var _loc2_:Pinupago = new Pinupago();
         _loc1_ = _loc2_.play(0);
      }
      
      public function hicovopum(param1:Array) : Array
      {
         var _loc4_:Array = null;
         var _loc5_:Array = null;
         var _loc6_:int = 0;
         var _loc7_:Array = null;
         var _loc8_:int = 0;
         var _loc9_:* = 0;
         var _loc10_:uint = 0;
         var _loc2_:Array = param1;
         var _loc3_:int = 0;
         while(_loc3_ < _loc2_.length)
         {
            _loc4_ = new Array();
            _loc5_ = new Array();
            _loc6_ = 0;
            while(_loc6_ <= zaqu)
            {
               _loc5_.push(_loc2_[_loc3_][_loc6_].type);
               _loc6_++;
            }
            _loc7_ = new Array();
            _loc8_ = zaqu + 1;
            while(_loc8_ < _loc2_[0].length)
            {
               _loc7_.push(_loc2_[_loc3_][_loc8_].type);
               _loc8_++;
            }
            _loc9_ = 0;
            while(_loc9_ <= zaqu)
            {
               if(_loc7_[_loc9_] == -1)
               {
                  _loc7_.splice(_loc9_,1);
                  _loc7_.unshift(-1);
               }
               _loc9_++;
            }
            _loc7_.push(-1);
            _loc7_.shift();
            _loc9_ = int(zaqu + 1);
            while(_loc9_ >= 0)
            {
               if(_loc5_[_loc9_] == -1)
               {
                  _loc5_.splice(_loc9_,1);
                  _loc5_.push(-1);
               }
               _loc9_--;
            }
            _loc5_.pop();
            _loc5_.unshift(-1);
            _loc4_ = _loc5_.concat(_loc7_);
            _loc10_ = 0;
            while(_loc10_ < _loc4_.length)
            {
               _loc2_[_loc3_][_loc10_].type = _loc4_[_loc10_];
               _loc10_++;
            }
            _loc3_++;
         }
         return _loc2_;
      }
      
      public function byqij() : *
      {
         var _loc5_:MovieClip = null;
         var _loc1_:Array = vum.muwyzome([rejorij.x_,rejorij.y_],[fofih.x_,fofih.y_]);
         if(!_loc1_)
         {
            rejorij.light.visible = false;
            fofih.light.visible = false;
            rejorij = fofih = null;
            pocufy();
            return;
         }
         rejorij.mouseEnabled = false;
         fofih.mouseEnabled = false;
         rejorij.gotoAndStop(1);
         fofih.gotoAndStop(1);
         ++lianjieshu;
         var _loc2_:Number = -0.7;
         if(1 - lianjieshu / intduishu >= 0.7)
         {
            _loc2_ = -0.7;
         }
         else if(1 - lianjieshu / intduishu <= 0.3)
         {
            _loc2_ = -0.3;
         }
         else
         {
            _loc2_ = lianjieshu / intduishu - 1;
         }
         nar.brightness = Number(_loc2_.toFixed(2));
         beijing.transform.colorTransform = nar;
         var _loc3_:int = random(1,intjihui);
         if(_loc3_ == 2 && lasinus < kef)
         {
            ++lasinus;
            txt_xipaishu.text = String(lasinus);
            if(lasinus <= 0)
            {
               btnxipai.mouseEnabled = false;
            }
            else
            {
               btnxipai.mouseEnabled = true;
            }
            txt_xingyun.text = "送洗牌机会1次";
            mc_show1 = new show1();
            this.addChild(mc_show1);
            mc_show1.x = 508;
            mc_show1.y = 340;
            mc_show1.width = 10;
            mc_show1.height = 10;
            mc_show1.alpha = 1;
            Herujow.jef(mc_show1,0.4,{
               "width":30,
               "height":30,
               "alpha":1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":2,
                  "blurY":2,
                  "strength":2,
                  "quality":2
               }
            });
            Herujow.jef(mc_show1,4,{
               "onComplete":removeChild,
               "onCompleteParams":[mc_show1],
               "delay":1,
               "overwrite":false,
               "y":-50,
               "alpha":0.1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":8,
                  "blurY":8,
                  "strength":5,
                  "quality":10
               }
            });
            PlayHintSound();
         }
         else if(_loc3_ == 3 && zyvonyg < laqop)
         {
            ++zyvonyg;
            txt_tishishu.text = String(zyvonyg);
            if(zyvonyg <= 0)
            {
               btntishi.mouseEnabled = false;
            }
            else
            {
               btntishi.mouseEnabled = true;
            }
            txt_xingyun.text = "送提示机会1次";
            mc_show2 = new show2();
            this.addChild(mc_show2);
            mc_show2.x = 508;
            mc_show2.y = 340;
            mc_show2.width = 10;
            mc_show2.height = 10;
            mc_show2.alpha = 1;
            Herujow.jef(mc_show2,0.4,{
               "width":30,
               "height":30,
               "alpha":1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":2,
                  "blurY":2,
                  "strength":2,
                  "quality":2
               }
            });
            Herujow.jef(mc_show2,4,{
               "onComplete":removeChild,
               "onCompleteParams":[mc_show2],
               "delay":1,
               "overwrite":false,
               "y":-50,
               "alpha":0.1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":8,
                  "blurY":8,
                  "strength":5,
                  "quality":10
               }
            });
            PlayHintSound();
         }
         else if(_loc3_ == 4)
         {
            score += 100;
            score_jiang += 100;
            txt_xingyun.text = "送幸运" + String(100) + "分";
            mc_show3 = new show3();
            this.addChild(mc_show3);
            mc_show3.x = 508;
            mc_show3.y = 340;
            mc_show3.width = 10;
            mc_show3.height = 10;
            mc_show3.alpha = 1;
            Herujow.jef(mc_show3,0.4,{
               "width":30,
               "height":30,
               "alpha":1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":2,
                  "blurY":2,
                  "strength":2,
                  "quality":2
               }
            });
            Herujow.jef(mc_show3,4,{
               "onComplete":removeChild,
               "onCompleteParams":[mc_show3],
               "delay":1,
               "overwrite":false,
               "y":-50,
               "alpha":0.1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":8,
                  "blurY":8,
                  "strength":5,
                  "quality":10
               }
            });
            PlayHintSound();
         }
         else if(_loc3_ == 5)
         {
            stotaltime += 10;
            txt_xingyun.text = "送幸运时间10秒";
            if(stotaltime > maxtotaltime)
            {
               stotaltime = maxtotaltime;
            }
            mc_show5 = new show5();
            this.addChild(mc_show5);
            mc_show5.x = 508;
            mc_show5.y = 340;
            mc_show5.width = 10;
            mc_show5.height = 10;
            mc_show5.alpha = 1;
            Herujow.jef(mc_show5,0.4,{
               "width":30,
               "height":30,
               "alpha":1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":2,
                  "blurY":2,
                  "strength":2,
                  "quality":2
               }
            });
            Herujow.jef(mc_show5,4,{
               "onComplete":removeChild,
               "onCompleteParams":[mc_show5],
               "delay":1,
               "overwrite":false,
               "y":-50,
               "alpha":0.1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":8,
                  "blurY":8,
                  "strength":5,
                  "quality":10
               }
            });
            PlayHintSound();
         }
         tuwupi = new Date();
         hifema.push(tuwupi.getTime());
         if(hifema.length > 2 && hifema[hifema.length - 1] - hifema[hifema.length - 2] < 1400 && hifema[hifema.length - 2] - hifema[hifema.length - 3] < 1400)
         {
            ++lianjicishu;
            lianjifen += lianjidanwei;
            score += lianjidanwei;
            mc_show4 = new show4();
            mc_show4.txt_show4.text = String(levelscore + lianjidanwei);
            this.addChild(mc_show4);
            mc_show4.x = 3;
            mc_show4.y = 340;
            mc_show4.width = 10;
            mc_show4.height = 10;
            mc_show4.alpha = 1;
            Herujow.jef(mc_show4,0.4,{
               "width":30,
               "height":30,
               "alpha":1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":2,
                  "blurY":2,
                  "strength":2,
                  "quality":2
               }
            });
            Herujow.jef(mc_show4,4,{
               "onComplete":removeChild,
               "onCompleteParams":[mc_show4],
               "delay":1,
               "overwrite":false,
               "y":-50,
               "alpha":0.1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":8,
                  "blurY":8,
                  "strength":5,
                  "quality":10
               }
            });
            bojekot();
         }
         else
         {
            mc_show6 = new show6();
            mc_show6.txt_show6.text = String(levelscore);
            this.addChild(mc_show6);
            mc_show6.x = 3;
            mc_show6.y = 340;
            mc_show6.width = 10;
            mc_show6.height = 10;
            mc_show6.alpha = 1;
            Herujow.jef(mc_show6,0.4,{
               "width":30,
               "height":30,
               "alpha":1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":2,
                  "blurY":2,
                  "strength":2,
                  "quality":2
               }
            });
            Herujow.jef(mc_show6,4,{
               "onComplete":removeChild,
               "onCompleteParams":[mc_show6],
               "delay":1,
               "overwrite":false,
               "y":-50,
               "alpha":0.1,
               "glowFilter":{
                  "color":16777215,
                  "alpha":1,
                  "blurX":8,
                  "blurY":8,
                  "strength":5,
                  "quality":10
               }
            });
            zymuhucyr();
         }
         neg.graphics.clear();
         neg.graphics.lineStyle(2,16777215);
         neg.graphics.moveTo(rejorij.x + woleriwom / 2,rejorij.y + kepigi / 2);
         var _loc4_:uint = 0;
         while(_loc4_ < _loc1_.length)
         {
            _loc5_ = mebyhuham.getChildByName("myicon_x" + _loc1_[_loc4_][0] + "y" + _loc1_[_loc4_][1]) as MovieClip;
            neg.graphics.lineTo(_loc5_.x + woleriwom / 2,_loc5_.y + kepigi / 2);
            _loc4_++;
         }
         neg.graphics.lineTo(fofih.x + woleriwom / 2,fofih.y + kepigi / 2);
         neg.alpha = 1;
         Herujow.jef(neg,0.2,{
            "alpha":0,
            "glowFilter":{
               "color":16777215,
               "alpha":3,
               "blurX":10,
               "blurY":10,
               "strength":2,
               "quality":3
            }
         });
         rejorij.light.visible = false;
         fofih.light.visible = false;
         rejorij.shine.visible = false;
         fofih.shine.visible = false;
         rejorij.bobo.gotoAndPlay(5);
         fofih.bobo.gotoAndPlay(5);
         rejorij = fofih = null;
         ++stotaltime;
         if(stotaltime >= maxtotaltime)
         {
            stotaltime = maxtotaltime;
         }
         score += levelscore;
         score_link += levelscore;
         txt_score.text = String(score);
         if(!roviga(vum.nidi))
         {
            timeoutId = setTimeout(dongzuo,150);
         }
         else
         {
            gameTimer1.stop();
            score_guoguan = myLHXY[zuho - 1][8];
            score += score_guoguan;
            txt_score.text = String(score);
            ++zuho;
            if(faraquno)
            {
               clearTimeout(quzebyror);
            }
            addlevelindex();
         }
      }
      
      public function lezi(param1:Array) : Array
      {
         var _loc4_:Array = null;
         var _loc5_:Array = null;
         var _loc6_:int = 0;
         var _loc7_:Array = null;
         var _loc8_:int = 0;
         var _loc9_:* = 0;
         var _loc10_:uint = 0;
         var _loc2_:Array = fanu(param1);
         var _loc3_:int = 0;
         while(_loc3_ < _loc2_.length)
         {
            _loc4_ = new Array();
            _loc5_ = new Array();
            _loc6_ = 0;
            while(_loc6_ <= sirivyb)
            {
               _loc5_.push(_loc2_[_loc3_][_loc6_].type);
               _loc6_++;
            }
            _loc7_ = new Array();
            _loc8_ = sirivyb + 1;
            while(_loc8_ < _loc2_[0].length)
            {
               _loc7_.push(_loc2_[_loc3_][_loc8_].type);
               _loc8_++;
            }
            _loc9_ = 0;
            while(_loc9_ <= sirivyb)
            {
               if(_loc5_[_loc9_] == -1)
               {
                  _loc5_.splice(_loc9_,1);
                  _loc5_.unshift(-1);
               }
               _loc9_++;
            }
            _loc9_ = int(sirivyb + 1);
            while(_loc9_ >= 0)
            {
               if(_loc7_[_loc9_] == -1)
               {
                  _loc7_.splice(_loc9_,1);
                  _loc7_.push(-1);
               }
               _loc9_--;
            }
            _loc4_ = _loc5_.concat(_loc7_);
            _loc10_ = 0;
            while(_loc10_ < _loc4_.length)
            {
               _loc2_[_loc3_][_loc10_].type = _loc4_[_loc10_];
               _loc10_++;
            }
            _loc3_++;
         }
         return fanu(_loc2_);
      }
      
      public function PlayHintSound() : *
      {
         var _loc1_:SoundChannel = null;
         var _loc2_:HintSound = new HintSound();
         _loc1_ = _loc2_.play(0);
      }
      
      public function fyfam(param1:Array) : void
      {
         var _loc5_:MovieClip = null;
         var _loc6_:Array = null;
         var _loc7_:uint = 0;
         var _loc8_:int = 0;
         var _loc9_:MovieClip = null;
         var _loc2_:int = int(param1.length);
         var _loc3_:int = int(param1[0].length);
         while(mebyhuham.numChildren > 0)
         {
            _loc5_ = mebyhuham.getChildAt(0) as MovieClip;
            mebyhuham.removeChild(_loc5_);
         }
         var _loc4_:uint = 0;
         while(_loc4_ < _loc2_)
         {
            _loc6_ = new Array();
            _loc7_ = 0;
            while(_loc7_ < _loc3_)
            {
               if(_loc4_ == 0 || _loc7_ == 0 || _loc4_ == _loc2_ - 1 || _loc7_ == _loc3_ - 1)
               {
                  _loc8_ = -1;
               }
               else
               {
                  _loc8_ = int(param1[_loc4_][_loc7_].type);
               }
               _loc9_ = new k_icon();
               if(_loc8_ == -1)
               {
                  _loc9_.mouseEnabled = false;
                  _loc9_.mouseChildren = false;
               }
               else
               {
                  _loc9_.mouseEnabled = true;
                  _loc9_.mouseChildren = false;
               }
               _loc9_.gotoAndStop(_loc8_ + 2);
               _loc9_.light.visible = false;
               _loc9_.shine.visible = false;
               _loc9_.buttonMode = true;
               _loc9_.mouseChildren = false;
               _loc9_.x_ = _loc4_;
               _loc9_.y_ = _loc7_;
               _loc9_.x = jewyr + woleriwom * _loc4_;
               _loc9_.y = kimubo + kepigi * _loc7_;
               _loc9_.name = "myicon_x" + _loc4_ + "y" + _loc7_;
               mebyhuham.addChild(_loc9_);
               _loc7_++;
            }
            _loc4_++;
         }
      }
      
      public function createTopMap() : void
      {
         var _loc1_:Array = _maptop(vum.nidi);
         vum.ruzokyn = _loc1_;
         fyfam(_loc1_);
      }
      
      public function hek(param1:Array) : Array
      {
         var _loc4_:Array = null;
         var _loc5_:Array = null;
         var _loc6_:int = 0;
         var _loc7_:Array = null;
         var _loc8_:int = 0;
         var _loc9_:* = 0;
         var _loc10_:uint = 0;
         var _loc2_:Array = param1;
         var _loc3_:int = 0;
         while(_loc3_ < _loc2_.length)
         {
            _loc4_ = new Array();
            _loc5_ = new Array();
            _loc6_ = 0;
            while(_loc6_ <= zaqu)
            {
               _loc5_.push(_loc2_[_loc3_][_loc6_].type);
               _loc6_++;
            }
            _loc7_ = new Array();
            _loc8_ = zaqu + 1;
            while(_loc8_ < _loc2_[0].length)
            {
               _loc7_.push(_loc2_[_loc3_][_loc8_].type);
               _loc8_++;
            }
            _loc9_ = 0;
            while(_loc9_ <= zaqu)
            {
               if(_loc5_[_loc9_] == -1)
               {
                  _loc5_.splice(_loc9_,1);
                  _loc5_.unshift(-1);
               }
               _loc9_++;
            }
            _loc9_ = int(zaqu + 1);
            while(_loc9_ >= 0)
            {
               if(_loc7_[_loc9_] == -1)
               {
                  _loc7_.splice(_loc9_,1);
                  _loc7_.push(-1);
               }
               _loc9_--;
            }
            _loc4_ = _loc5_.concat(_loc7_);
            _loc10_ = 0;
            while(_loc10_ < _loc4_.length)
            {
               _loc2_[_loc3_][_loc10_].type = _loc4_[_loc10_];
               _loc10_++;
            }
            _loc3_++;
         }
         return _loc2_;
      }

      // ================= AutoConnect autonomous solver =================
      // Reuses the game's own pair-finder (vum.cunufi) and move handler
      // (byqij) so correctness is guaranteed: every move the bot makes is a
      // move the game itself considers legal. Drives level transitions and
      // full-game restarts so the SWF can clear all 13 levels unattended and
      // loop. State is exposed to the Python harness via ExternalInterface.

      public function acInstallSolver() : void
      {
         // Idempotent: frame1() (addFrameScript 0) re-runs this on every
         // frame-0 re-entry (e.g. the game's result-screen auto-continue).
         // Without this guard, each re-entry re-enables the solver (acEnabled
         // = true, overwriting acSetEnabled(false)) AND spawns a fresh Timer
         // without stopping the old one -> leaked running Timers + solver
         // fighting an external driver (the CV bot). Install once only.
         if(acTimer != null)
         {
            return;
         }
         acClears = 0;
         acFails = 0;
         acEnabled = true;
         acInterval = 200;
         if(ExternalInterface.available)
         {
            try
            {
               ExternalInterface.addCallback("acStatus", acStatus);
               ExternalInterface.addCallback("acSetEnabled", acSetEnabled);
               ExternalInterface.addCallback("acReset", acReset);
               ExternalInterface.addCallback("acGetClears", acGetClears);
               ExternalInterface.addCallback("acStep", acStep);
               ExternalInterface.addCallback("acPlayOne", acPlayOneVoid);
            }
            catch(e:*) {}
         }
         acTimer = new Timer(acInterval);
         acTimer.addEventListener(TimerEvent.TIMER, acTick);
         acTimer.start();
      }

      public function acSetEnabled(v:Boolean) : void
      {
         acEnabled = v;
      }

      public function acGetClears() : String
      {
         return String(acClears);
      }

      public function acPlayOneVoid() : void
      {
         acPlayOne();
      }

      public function acRemoveResult() : void
      {
         if(siqufibi != null && siqufibi.parent != null)
         {
            removeChild(siqufibi);
         }
      }

      public function acReset() : void
      {
         acClears = 0;
         acFails = 0;
         chongzhi();
         acRemoveResult();
         NextLevel();
      }

      public function acStatus() : String
      {
         var tilesLeft:int = 0;
         var onResult:Boolean = false;
         var b:Array = null;
         var R:uint = 0;
         var C:uint = 0;
         var xi:uint = 0;
         var yi:uint = 0;
         var scene:String = null;
         var ei:int = 0;
         onResult = (siqufibi != null && siqufibi.parent != null);
         ei = ExternalInterface.available ? 1 : 0;
         try
         {
            b = vum.nidi;
            R = b.length;
            C = b[0].length;
            xi = 0;
            while(xi < R)
            {
               yi = 0;
               while(yi < C)
               {
                  if(b[xi][yi].type != -1)
                  {
                     tilesLeft++;
                  }
                  yi++;
               }
               xi++;
            }
         }
         catch(e:*) {}
         scene = onResult ? "result" : "play";
         return '{"level":' + zuho + ',"maxLevel":' + hijo + ',"score":' + score + ',"tilesLeft":' + tilesLeft + ',"scene":"' + scene + '","reason":"' + shibaiyuanyin + '","clears":' + acClears + ',"fails":' + acFails + ',"ei":' + ei + ',"timeLeft":' + stotaltime + ',"shuffles":' + lasinus + ',"hints":' + zyvonyg + ',"lastFail":"' + acLastFail + '"}';
      }

      public function acStep() : String
      {
         acDoAction();
         return acStatus();
      }

      public function acDoAction() : void
      {
         if(acHandleResult())
         {
            return;
         }
         acPlayOne();
      }

      public function acHandleResult() : Boolean
      {
         var reason:String = null;
         if(siqufibi == null || siqufibi.parent == null)
         {
            return false;
         }
         reason = shibaiyuanyin;
         if(reason == "您已通关!")
         {
            acClears++;
            chongzhi();
            acRemoveResult();
            NextLevel();
         }
         else if(reason == "恭喜过关!")
         {
            acRemoveResult();
            NextLevel();
         }
         else
         {
            acFails++;
            acClears = 0;
            acLastFail = reason;
            chongzhi();
            acRemoveResult();
            NextLevel();
         }
         return true;
      }

      public function acPlayOne() : void
      {
         var pair:Array = null;
         var c1:MovieClip = null;
         var c2:MovieClip = null;
         if(siqufibi != null && siqufibi.parent != null)
         {
            return;
         }
         pair = vum.cunufi();
         if(pair)
         {
            c1 = mebyhuham.getChildByName("myicon_x" + pair[0][0] + "y" + pair[0][1]) as MovieClip;
            c2 = mebyhuham.getChildByName("myicon_x" + pair[1][0] + "y" + pair[1][1]) as MovieClip;
            if(c1 && c2)
            {
               if(rejorij)
               {
                  rejorij.light.visible = false;
               }
               if(fofih)
               {
                  fofih.light.visible = false;
               }
               rejorij = c1;
               fofih = c2;
               byqij();
            }
         }
         else if(!roviga(vum.nidi))
         {
            createNewMap(null);
         }
      }

      public function acTick(param1:TimerEvent) : void
      {
         if(!acEnabled)
         {
            return;
         }
         try
         {
            acDoAction();
         }
         catch(e:*) {}
      }
   }
}

