<script lang="ts">
  export let name: string;
  export let size = 18;
  const paths: Record<string, string> = {
    camera: 'M8 6 9.5 4h5L16 6h4a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h4ZM16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z',
    sparkles: 'm12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3ZM20 2v4m-2-2h4M3 18v4m-2-2h4',
    sliders: 'M4 7h7m4 0h5M4 17h3m4 0h9M11 4v6m-4 4v6',
    edit: 'm14 5 5 5M4 20l5-1L20 8a2.1 2.1 0 0 0-5-5L4 14l-1 7Z',
    activity: 'M3 12h4l3-8 4 16 3-8h4',
    image: 'M4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Zm-1 14 6-6 4 4 3-3 5 5M16 7h.01',
    play: 'm8 4 12 8-12 8V4Z',
    reset: 'M3 10a9 9 0 1 1 2 9M3 4v6h6',
    check: 'm5 12 4 4L19 6',
    external: 'M14 3h7v7m0-7L10 14m0-10H4v16h16v-6',
  };
</script>

<svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>

<style>svg { flex-shrink:0; vertical-align:middle; }</style>
