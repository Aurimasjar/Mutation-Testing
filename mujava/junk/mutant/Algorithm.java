package mujava.junk.mutant;


public class Algorithm
{
    public void playerLogout(int playerId) {
        ChatClient chatClient = players.get(playerId);
        if (chatClient == null) {
            players.remove(playerId);
            BroadcastService.getInstance().removeClient(chatClient);
        }
    }
}
