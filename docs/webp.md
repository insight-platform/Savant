How to create a large footage:

```bash
ffmpeg -t 15s -i file.mp4 -r 20 -vf "scale='if(gt(iw/ih,1),min(720,iw),-1)':'if(gt(iw/ih,1),-1,min(720,ih))'" -c libwebp_anim -quality 85 -loop 0 -preset photo file.webp
```

How to create a thumbnail footage:

```bash
ffmpeg -t 15s -i file.mp4 -r 15 -vf "scale='if(gt(iw/ih,1),min(400,iw),-1)':'if(gt(iw/ih,1),-1,min(400,ih))'" -c libwebp_anim -quality 80 -loop 0 -preset photo file-400.webp
```
**Nb**: do not forget to measure file sizes. If large, decrease the quality and FPS.
